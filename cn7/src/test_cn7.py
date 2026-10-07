from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config_cn7 import (
    CV_SEEDS,
    FEATURE_COLS,
    HIGH_PRECISION_THRESHOLD,
    HIGH_RECALL_THRESHOLD,
    MODEL_DIR,
    OPERATION_THRESHOLD,
    ROLL_MEAN_COLS,
    ROLL_WINDOW,
    SENSOR_COLS,
    SESSION_GAP_SECONDS,
    SIDE_FEATURE,
    TIME_COL,
    SHARED_RAW_DIR,
)
from .ood_cn7 import load_ood_artifact, score_ood_cycles


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_SENSOR_PREFIX = "Raw_"

TEST_MOLDSET_PATH = ROOT_DIR / "data" / "raw" / "moldset_unlabeled_cn7.csv"
TEST_RAW_PATH = SHARED_RAW_DIR / "unlabeled_data.csv"

TEST_PREPROCESSED_PATH = ROOT_DIR / "data" / "processed" / "test_cn7_preprocessed.csv"
TEST_MODEL_INPUT_PATH = ROOT_DIR / "data" / "processed" / "cn7_test_model_input.csv"
TEST_INFERENCE_PATH = ROOT_DIR / "outputs" / "cn7_test_inference_result.csv"
TEST_FLAGGED_PATH = ROOT_DIR / "outputs" / "cn7_test_flagged_only.csv"
TEST_SUMMARY_PATH = ROOT_DIR / "outputs" / "cn7_test_summary.csv"
TEST_OOD_CYCLE_PATH = ROOT_DIR / "outputs" / "cn7_test_ood_cycle_result.csv"


def restore_test_metadata(scaled_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    moldset_unlabeled_cn7.csv의 scaled 센서값은 supervised model용으로 유지하고,
    unlabeled_data.csv의 원시 센서값은 Raw_<sensor> 컬럼으로 함께 복원한다.

    PCA OOD는 raw 센서 공간에 RobustScaler를 적용해 학습했으므로 반드시
    Raw_<sensor>를 사용해야 한다.
    """
    scaled = scaled_df.copy()
    raw = raw_df.copy()
    scaled.columns = scaled.columns.astype(str).str.strip()
    raw.columns = raw.columns.astype(str).str.strip()

    if "Unnamed: 0" not in scaled.columns:
        raise ValueError("moldset_unlabeled_cn7.csv에 Unnamed: 0 컬럼이 없습니다.")
    if TIME_COL not in raw.columns:
        raise ValueError("unlabeled_data.csv에 TimeStamp 컬럼이 없습니다.")
    if "PART_NAME" not in raw.columns:
        raise ValueError("unlabeled_data.csv에 PART_NAME 컬럼이 없습니다.")

    missing_raw_sensors = [c for c in SENSOR_COLS if c not in raw.columns]
    if missing_raw_sensors:
        raise ValueError(f"unlabeled_data.csv 원시 센서 컬럼 누락: {missing_raw_sensors}")

    original_index = pd.to_numeric(scaled["Unnamed: 0"], errors="coerce")
    if original_index.isna().any():
        raise ValueError("Unnamed: 0에 숫자가 아닌 값이 있습니다.")
    original_index = original_index.astype(int)

    if original_index.min() < 0 or original_index.max() >= len(raw):
        raise IndexError("원본 index가 unlabeled_data.csv 범위를 벗어납니다.")

    matched = raw.iloc[original_index.to_numpy()].reset_index(drop=True)

    if "Unnamed: 0" in matched.columns:
        raw_index = pd.to_numeric(matched["Unnamed: 0"], errors="coerce")
        if not np.array_equal(
            raw_index.to_numpy(dtype=int),
            original_index.to_numpy(dtype=int),
        ):
            raise ValueError(
                "moldset_unlabeled_cn7.csv와 unlabeled_data.csv의 index가 일치하지 않습니다."
            )

    cn7_mask = matched["PART_NAME"].astype(str).str.contains("CN7", case=False, na=False)
    if not cn7_mask.all():
        raise ValueError(f"CN7이 아닌 행이 {(~cn7_mask).sum()}건 있습니다.")

    restored = scaled.reset_index(drop=True).copy()
    restored["original_index"] = original_index.to_numpy()
    restored[TIME_COL] = pd.to_datetime(matched[TIME_COL], errors="coerce")
    restored["PART_NAME"] = matched["PART_NAME"].to_numpy()
    if restored[TIME_COL].isna().any():
        raise ValueError("TimeStamp 변환 실패 행이 있습니다.")

    part = restored["PART_NAME"].astype(str).str.upper()
    rh = part.str.contains(r"\bRH\b", regex=True, na=False)
    lh = part.str.contains(r"\bLH\b", regex=True, na=False)
    if (~(rh | lh)).any():
        raise ValueError("PART_NAME에서 RH/LH를 판별할 수 없습니다.")
    restored[SIDE_FEATURE] = rh.astype(int)

    for col in [
        "PART_FACT_PLAN_DATE",
        "PART_FACT_SERIAL",
        "PART_NO",
        "EQUIP_CD",
        "EQUIP_NAME",
    ]:
        if col in matched.columns:
            restored[col] = matched[col].to_numpy()

    # scaled 센서: LightGBM용
    missing = [c for c in SENSOR_COLS if c not in restored.columns]
    if missing:
        raise ValueError(f"테스트 scaled 센서 컬럼 누락: {missing}")
    for col in SENSOR_COLS:
        restored[col] = pd.to_numeric(restored[col], errors="coerce")

    # raw 센서: PCA OOD용
    for col in SENSOR_COLS:
        restored[f"{RAW_SENSOR_PREFIX}{col}"] = pd.to_numeric(
            matched[col], errors="coerce"
        ).to_numpy()

    scaled_bad = restored[SENSOR_COLS].isna().any()
    if scaled_bad.any():
        raise ValueError(f"scaled 센서 결측/숫자 변환 문제: {scaled_bad[scaled_bad].index.tolist()}")

    raw_cols = [f"{RAW_SENSOR_PREFIX}{c}" for c in SENSOR_COLS]
    raw_bad = restored[raw_cols].isna().any()
    if raw_bad.any():
        raise ValueError(f"raw 센서 결측/숫자 변환 문제: {raw_bad[raw_bad].index.tolist()}")

    return restored.sort_values([TIME_COL, "original_index"]).reset_index(drop=True)


def add_test_cycle_id(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    sensor_hash = pd.util.hash_pandas_object(work[SENSOR_COLS], index=False).astype(str)
    work["_cycle_key"] = work[TIME_COL].astype(str) + "__" + sensor_hash

    cycle_order = (
        work.groupby("_cycle_key", sort=False)
        .agg(
            _cycle_time=(TIME_COL, "first"),
            _cycle_original_index=("original_index", "min"),
        )
        .reset_index()
        .sort_values(["_cycle_time", "_cycle_original_index"])
        .reset_index(drop=True)
    )
    cycle_order["_cycle_id"] = np.arange(len(cycle_order))
    return work.merge(
        cycle_order[["_cycle_key", "_cycle_id"]],
        on="_cycle_key",
        how="left",
        validate="many_to_one",
    )


def create_test_features(df: pd.DataFrame) -> pd.DataFrame:
    work = add_test_cycle_id(df)

    cycle = (
        work[["_cycle_id", TIME_COL, "original_index"] + SENSOR_COLS]
        .sort_values(["_cycle_id", "original_index"])
        .drop_duplicates(subset=["_cycle_id"], keep="first")
        .sort_values("_cycle_id")
        .reset_index(drop=True)
    )

    cycle["_time_gap_sec"] = cycle[TIME_COL].diff().dt.total_seconds()
    cycle["_session_id"] = cycle["_time_gap_sec"].gt(SESSION_GAP_SECONDS).cumsum()

    for sensor, roll_col in zip(SENSOR_COLS, ROLL_MEAN_COLS):
        cycle[roll_col] = (
            cycle.groupby("_session_id")[sensor]
            .transform(
                lambda s: s.shift(1).rolling(window=ROLL_WINDOW, min_periods=1).mean()
            )
            .fillna(0)
        )

    result = work.merge(
        cycle[["_cycle_id", "_session_id"] + ROLL_MEAN_COLS],
        on="_cycle_id",
        how="left",
        validate="many_to_one",
    )

    if result[ROLL_MEAN_COLS].isna().any().any():
        raise ValueError("Rolling Mean 생성 후 결측치가 있습니다.")

    missing = [c for c in FEATURE_COLS if c not in result.columns]
    if missing:
        raise ValueError(f"최종 Feature 누락: {missing}")
    if len(FEATURE_COLS) != 47:
        raise RuntimeError(f"Feature 수가 47이 아닙니다: {len(FEATURE_COLS)}")

    return result


def load_models():
    artifacts = []
    for seed in CV_SEEDS:
        path = MODEL_DIR / f"cn7_lgbm_seed{seed}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {path}\n먼저 python run_cn7.py 를 실행하세요."
            )
        artifacts.append(joblib.load(path))
    return artifacts


def get_risk_level(prob: float) -> str:
    if prob >= HIGH_PRECISION_THRESHOLD:
        return "CRITICAL"
    if prob >= OPERATION_THRESHOLD:
        return "HIGH"
    if prob >= HIGH_RECALL_THRESHOLD:
        return "WATCH"
    return "LOW"


def get_priority(row) -> str:
    # OOD는 기존 failure classifier가 모르는 신규 공정상태이므로 최우선 안전계층으로 둔다.
    if row.get("OOD_Flag", 0) == 1:
        return "P0_OOD"
    if row["Risk_Probability"] >= HIGH_PRECISION_THRESHOLD:
        return "P1_CRITICAL"
    if row["Risk_Probability"] >= OPERATION_THRESHOLD:
        return "P2_HIGH"
    if row["Prediction_Stability"] == "Unstable":
        return "P3_UNCERTAIN"
    return "P4_NORMAL"


def _build_ood_cycle_result(feature_df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [f"{RAW_SENSOR_PREFIX}{c}" for c in SENSOR_COLS]
    missing = [c for c in raw_cols if c not in feature_df.columns]
    if missing:
        raise ValueError(f"OOD raw sensor 컬럼 누락: {missing}")

    cycle = (
        feature_df[["_cycle_id", "_session_id", TIME_COL] + raw_cols]
        .sort_values(["_cycle_id", TIME_COL])
        .drop_duplicates("_cycle_id")
        .copy()
    )
    rename = {f"{RAW_SENSOR_PREFIX}{c}": c for c in SENSOR_COLS}
    score_input = cycle.rename(columns=rename)

    artifact = load_ood_artifact()
    scored = score_ood_cycles(score_input, artifact)

    # 세션 전체가 baseline과 다른 경우 개별 OOD를 '불량 확정'으로 해석하지 않도록 drift 상태를 별도 표시한다.
    p0 = float(artifact.get("calibration_ood_rate", 0.01))
    if p0 <= 0:
        p0 = 1.0 / max(int(artifact.get("calibration_cycle_count", 100)), 1)

    session_rows = []
    for session_id, g in scored.groupby("_session_id"):
        applicable = g[g["OOD_Applicable"] == 1]
        n = len(applicable)

        if n == 0:
            rate = np.nan
            ucl = np.nan
            drift_flag = 0
        else:
            rate = float(applicable["OOD_Flag"].mean())
            ucl = min(1.0, p0 + 3 * np.sqrt(p0 * (1 - p0) / max(n, 1)))
            drift_flag = int(rate > ucl)

        session_rows.append(
            {
                "_session_id": session_id,
                "Session_Cycle_Count": len(g),
                "Session_OOD_Applicable_Cycles": n,
                "Session_OOD_Rate": rate,
                "Session_Drift_UCL": ucl,
                "Session_Drift_Flag": drift_flag,
            }
        )

    session_df = pd.DataFrame(session_rows)
    scored = scored.merge(session_df, on="_session_id", how="left", validate="many_to_one")
    return scored


def predict_test(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = load_models()
    x = feature_df[FEATURE_COLS].copy()

    probabilities = [a["model"].predict_proba(x)[:, 1] for a in artifacts]
    matrix = np.column_stack(probabilities)
    risk = matrix.mean(axis=1)
    risk_std = matrix.std(axis=1)
    votes = (matrix >= OPERATION_THRESHOLD).sum(axis=1)
    stable = (votes == 0) | (votes == len(CV_SEEDS))

    result_cols = [
        "original_index",
        TIME_COL,
        "PART_NAME",
        SIDE_FEATURE,
        "_cycle_id",
        "_session_id",
    ]
    for col in ["PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NO", "EQUIP_CD", "EQUIP_NAME"]:
        if col in feature_df.columns:
            result_cols.append(col)

    result = feature_df[result_cols].copy()
    for seed, proba in zip(CV_SEEDS, probabilities):
        result[f"Proba_Model_Seed_{seed}"] = proba

    result["Risk_Probability"] = risk
    # 표시용 0~100 점수. 0.060 probability threshold는 6.0 score와 동일하다.
    result["Risk_Score"] = risk * 100
    result["Risk_Threshold_Probability"] = OPERATION_THRESHOLD
    result["Risk_Threshold_Score"] = OPERATION_THRESHOLD * 100
    result["Risk_STD"] = risk_std
    result["Positive_Votes"] = votes
    result["Prediction_Stability"] = np.where(stable, "Stable", "Unstable")
    result["Risk_Level"] = [get_risk_level(p) for p in risk]
    result["Risk_Only_Flag"] = (risk >= OPERATION_THRESHOLD).astype(int)
    result["Uncertainty_Flag"] = (~stable).astype(int)
    result["Risk_Uncertainty_Flag"] = (
        (result["Risk_Only_Flag"] == 1) | (result["Uncertainty_Flag"] == 1)
    ).astype(int)

    ood_cycle = _build_ood_cycle_result(feature_df)
    result = result.merge(
        ood_cycle[
            [
                "_cycle_id",
                "OOD_Score",
                "OOD_Threshold",
                "OOD_Applicable",
                "OOD_Flag",
                "Session_OOD_Rate",
                "Session_Drift_UCL",
                "Session_Drift_Flag",
            ]
        ],
        on="_cycle_id",
        how="left",
        validate="many_to_one",
    )

    result["Final_Inspection_Flag"] = (
        (result["Risk_Uncertainty_Flag"] == 1) | (result["OOD_Flag"] == 1)
    ).astype(int)
    result["Model_Trust_Status"] = np.select(
        [
            result["OOD_Applicable"] == 0,
            result["Session_Drift_Flag"] == 1,
        ],
        [
            "OOD_NOT_APPLICABLE_PRE_BASELINE",
            "DEGRADED_BY_DRIFT",
        ],
        default="NORMAL",
    )

    result["Inspection_Priority"] = result.apply(get_priority, axis=1)
    priority_order = {
        "P0_OOD": 0,
        "P1_CRITICAL": 1,
        "P2_HIGH": 2,
        "P3_UNCERTAIN": 3,
        "P4_NORMAL": 4,
    }
    action_map = {
        "P0_OOD": "OOD 신규 공정상태 우선 검사",
        "P1_CRITICAL": "즉시 검사",
        "P2_HIGH": "우선 검사",
        "P3_UNCERTAIN": "불확실성 추가 검사",
        "P4_NORMAL": "일반 검사",
    }
    result["_priority_order"] = result["Inspection_Priority"].map(priority_order)
    result["Inspection_Action"] = result["Inspection_Priority"].map(action_map)

    drift_mask = result["Session_Drift_Flag"] == 1
    result.loc[drift_mask, "Inspection_Action"] = (
        result.loc[drift_mask, "Inspection_Action"]
        + " / Session Drift: 기존 품질검사 유지·모델 신뢰 하향"
    )

    result = result.sort_values(
        ["_priority_order", "OOD_Score", "Risk_Probability", "Risk_STD"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    result["Inspection_Rank"] = np.arange(1, len(result) + 1)
    result = result.drop(columns=["_priority_order"])

    return result, ood_cycle


def build_summary(result: pd.DataFrame, ood_cycle: pd.DataFrame) -> pd.DataFrame:
    total = len(result)
    risk_only_count = int(result["Risk_Only_Flag"].sum())
    risk_uncertainty_count = int(result["Risk_Uncertainty_Flag"].sum())
    final_count = int(result["Final_Inspection_Flag"].sum())
    priority_counts = result["Inspection_Priority"].value_counts().to_dict()

    rows = [
        {"Metric": "Total_Test_Rows", "Value": total},
        {"Metric": "Unique_Cycles", "Value": int(result["_cycle_id"].nunique())},
        {"Metric": "Risk_Only_Flag_Count", "Value": risk_only_count},
        {"Metric": "Risk_Only_Flag_Rate_pct", "Value": risk_only_count / total * 100},
        {"Metric": "Risk_Plus_Uncertainty_Count", "Value": risk_uncertainty_count},
        {"Metric": "Risk_Plus_Uncertainty_Rate_pct", "Value": risk_uncertainty_count / total * 100},
        {"Metric": "OOD_Cycle_Flag_Count", "Value": int(ood_cycle["OOD_Flag"].sum())},
        {"Metric": "OOD_Cycle_Flag_Rate_pct", "Value": float(ood_cycle["OOD_Flag"].mean() * 100)},
        {"Metric": "Session_Drift_Count", "Value": int(ood_cycle.loc[ood_cycle["Session_Drift_Flag"] == 1, "_session_id"].nunique())},
        {"Metric": "Final_Inspection_Count", "Value": final_count},
        {"Metric": "Final_Inspection_Rate_pct", "Value": final_count / total * 100},
        {"Metric": "P0_OOD", "Value": priority_counts.get("P0_OOD", 0)},
        {"Metric": "P1_CRITICAL", "Value": priority_counts.get("P1_CRITICAL", 0)},
        {"Metric": "P2_HIGH", "Value": priority_counts.get("P2_HIGH", 0)},
        {"Metric": "P3_UNCERTAIN", "Value": priority_counts.get("P3_UNCERTAIN", 0)},
        {"Metric": "P4_NORMAL", "Value": priority_counts.get("P4_NORMAL", 0)},
        {"Metric": "Risk_Probability_Mean", "Value": float(result["Risk_Probability"].mean())},
        {"Metric": "Risk_Probability_Max", "Value": float(result["Risk_Probability"].max())},
    ]
    return pd.DataFrame(rows)


def run_test_pipeline():
    for path in [TEST_MOLDSET_PATH, TEST_RAW_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"파일이 없습니다: {path}")

    print("\n[TEST 1/5] RESTORE METADATA + RAW SENSOR")
    scaled = pd.read_csv(TEST_MOLDSET_PATH)
    raw = pd.read_csv(TEST_RAW_PATH)
    restored = restore_test_metadata(scaled, raw)
    TEST_PREPROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    restored.to_csv(TEST_PREPROCESSED_PATH, index=False, encoding="utf-8-sig")
    print(f"Rows          : {len(restored):,}")
    print(f"Time range    : {restored[TIME_COL].min()} ~ {restored[TIME_COL].max()}")

    print("\n[TEST 2/5] FEATURE ENGINEERING")
    feature_df = create_test_features(restored)
    feature_df.to_csv(TEST_MODEL_INPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Records       : {len(feature_df):,}")
    print(f"Unique cycles : {feature_df['_cycle_id'].nunique():,}")
    print(f"Features      : {len(FEATURE_COLS)}")

    print("\n[TEST 3/5] LIGHTGBM + PCA OOD INFERENCE")
    result, ood_cycle = predict_test(feature_df)
    TEST_INFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(TEST_INFERENCE_PATH, index=False, encoding="utf-8-sig")
    ood_cycle.to_csv(TEST_OOD_CYCLE_PATH, index=False, encoding="utf-8-sig")

    print("\n[TEST 4/5] PRIORITY EXPORT")
    flagged = result[result["Final_Inspection_Flag"] == 1].copy()
    flagged.to_csv(TEST_FLAGGED_PATH, index=False, encoding="utf-8-sig")

    print("\n[TEST 5/5] SUMMARY")
    summary = build_summary(result, ood_cycle)
    summary.to_csv(TEST_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))

    print("\n저장 완료")
    print(f"- {TEST_INFERENCE_PATH}")
    print(f"- {TEST_OOD_CYCLE_PATH}")
    print(f"- {TEST_FLAGGED_PATH}")
    print(f"- {TEST_SUMMARY_PATH}")

    return result, summary
