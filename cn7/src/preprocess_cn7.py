from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config_cn7 import (
    RAW_LABELED_PATH,
    RESTORE_FEATURE_COLS,
    SENSOR_COLS,
    SIDE_FEATURE,
    TARGET,
    TIME_COL,
)


TRAIN_DATES = ["2020-10-16", "2020-10-19", "2020-10-20"]
EXPECTED_TRAIN_ROWS = 1211


def _normalize_target(series: pd.Series) -> pd.Series:
    """CN7 target을 정상=0, 불량=1로 통일한다."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        values = set(numeric.astype(int).unique())
        if values.issubset({0, 1}):
            return numeric.astype(int)

    text = series.astype(str).str.strip().str.upper()
    mapping = {
        "Y": 0,
        "N": 1,
        "PASS": 0,
        "FAIL": 1,
        "NORMAL": 0,
        "DEFECT": 1,
        "0": 0,
        "1": 1,
    }

    mapped = text.map(mapping)

    if mapped.isna().any():
        bad = sorted(text[mapped.isna()].unique().tolist())
        raise ValueError(f"{TARGET} 변환 실패 값: {bad}")

    return mapped.astype(int)


def _extract_side_from_part_name(series: pd.Series) -> pd.Series:
    part = series.astype(str).str.upper()

    rh = part.str.contains(r"\bRH\b", regex=True, na=False)
    lh = part.str.contains(r"\bLH\b", regex=True, na=False)

    side = pd.Series(index=series.index, dtype="object")
    side.loc[rh] = "RH"
    side.loc[lh] = "LH"

    if side.isna().any():
        bad = series[side.isna()].drop_duplicates().tolist()[:10]
        raise ValueError(
            "PART_NAME에서 RH/LH를 복원하지 못했습니다. "
            f"예시: {bad}"
        )

    return side


def _make_side_rh(df: pd.DataFrame) -> pd.Series:
    if SIDE_FEATURE in df.columns:
        side = pd.to_numeric(df[SIDE_FEATURE], errors="coerce")
        if side.notna().all() and set(side.astype(int).unique()).issubset({0, 1}):
            return side.astype(int)

    for col in ["RH_LH", "Side"]:
        if col in df.columns:
            text = df[col].astype(str).str.strip().str.upper()
            if text.isin(["RH", "LH"]).all():
                return (text == "RH").astype(int)

    if "PART_NAME" in df.columns:
        return (_extract_side_from_part_name(df["PART_NAME"]) == "RH").astype(int)

    raise ValueError(
        "Side 정보가 없습니다. Side_RH, RH_LH, Side 또는 PART_NAME이 필요합니다."
    )


def _prepare_raw_cn7_candidate(raw: pd.DataFrame) -> pd.DataFrame:
    """
    labeled_data.csv에서 moldset_labeled_cn7.csv에 대응하는
    CN7 학습 원본 1,211행을 추출한다.

    실제 원본 분포:
    - 2020-10-16: 375행
    - 2020-10-19: 72행
    - 2020-10-20: 764행
    - 합계: 1,211행
    """
    work = raw.copy()
    work.columns = work.columns.astype(str).str.strip()

    required = [TIME_COL, "PART_NAME", TARGET] + RESTORE_FEATURE_COLS
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(
            "labeled_data.csv에서 CN7 복원에 필요한 컬럼이 없습니다: "
            f"{missing}"
        )

    work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="coerce")
    if work[TIME_COL].isna().any():
        raise ValueError("labeled_data.csv의 TimeStamp 변환에 실패했습니다.")

    cn7_mask = work["PART_NAME"].astype(str).str.contains(
        "CN7",
        case=False,
        na=False,
    )

    date_mask = (
        work[TIME_COL]
        .dt.strftime("%Y-%m-%d")
        .isin(TRAIN_DATES)
    )

    work = work.loc[cn7_mask & date_mask].copy()

    # 원본 순서 보존. moldset_labeled_cn7.csv와 행 순서가 동일하다.
    work = work.reset_index().rename(columns={"index": "original_index"})

    date_counts = (
        work[TIME_COL]
        .dt.strftime("%Y-%m-%d")
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_counts = {
        "2020-10-16": 375,
        "2020-10-19": 72,
        "2020-10-20": 764,
    }

    if len(work) != EXPECTED_TRAIN_ROWS or date_counts != expected_counts:
        raise ValueError(
            "CN7 학습 원본 구성이 예상과 다릅니다. "
            f"현재 행 수={len(work):,}, 날짜별 행 수={date_counts}, "
            f"기대값={expected_counts}"
        )

    return work


def restore_cn7_metadata(
    scaled_df: pd.DataFrame,
    raw_labeled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    TimeStamp가 없는 moldset_labeled_cn7.csv에 원본 메타데이터를 복원한다.

    검증된 실제 관계:
    1) labeled_data.csv에서 CN7 + 2020-10-16/19/20 추출 → 1,211행
    2) 해당 1,211행의 24개 공정변수에 StandardScaler 적용
    3) moldset_labeled_cn7.csv의 24개 변수와 행 순서대로 비교
    4) 평균 오차 약 9e-14, 최대 오차 약 2.4e-13
    5) PassOrFail도 1,211행 모두 일치

    따라서 거리 기반 재배열(Hungarian matching)은 사용하지 않고
    검증된 원본 행 순서를 그대로 사용한다.
    """
    scaled = scaled_df.copy()
    scaled.columns = scaled.columns.astype(str).str.strip()

    missing_scaled = [c for c in RESTORE_FEATURE_COLS if c not in scaled.columns]
    if missing_scaled:
        raise ValueError(
            "moldset_labeled_cn7.csv에서 복원용 24개 공정변수가 누락되었습니다: "
            f"{missing_scaled}"
        )

    raw_candidate = _prepare_raw_cn7_candidate(raw_labeled_df)

    if len(scaled) != len(raw_candidate):
        raise ValueError(
            "scaled CN7과 원본 후보 행 수가 다릅니다. "
            f"scaled={len(scaled):,}, raw_candidate={len(raw_candidate):,}"
        )

    X_scaled = scaled[RESTORE_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    X_raw = raw_candidate[RESTORE_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")

    if X_scaled.isna().any().any():
        bad = X_scaled.columns[X_scaled.isna().any()].tolist()
        raise ValueError(f"scaled CN7 숫자 변환/결측 문제: {bad}")

    if X_raw.isna().any().any():
        bad = X_raw.columns[X_raw.isna().any()].tolist()
        raise ValueError(f"raw CN7 숫자 변환/결측 문제: {bad}")

    X_raw_z = StandardScaler().fit_transform(X_raw)
    X_scaled_np = X_scaled.to_numpy(dtype=float)

    row_distance = np.linalg.norm(
        X_scaled_np - X_raw_z,
        axis=1,
    )

    print("\n[CN7 METADATA RESTORE]")
    print(f"Matched rows : {len(raw_candidate):,}")
    print(f"Date counts  : {_prepare_date_counts(raw_candidate)}")
    print(f"Distance mean: {row_distance.mean():.6e}")
    print(f"Distance max : {row_distance.max():.6e}")

    # 실데이터 검증값은 약 1e-13. 넉넉하게 1e-6을 허용한다.
    if row_distance.max() > 1e-6:
        raise ValueError(
            "moldset_labeled_cn7.csv와 labeled_data.csv의 행 정렬/스케일링이 "
            "기대값과 다릅니다. 잘못된 TimeStamp 복원을 막기 위해 중단합니다. "
            f"mean={row_distance.mean():.6e}, max={row_distance.max():.6e}"
        )

    restored = scaled.reset_index(drop=True).copy()
    matched_raw = raw_candidate.reset_index(drop=True)

    restored["original_index"] = matched_raw["original_index"].to_numpy()
    restored[TIME_COL] = matched_raw[TIME_COL].to_numpy()
    restored["PART_NAME"] = matched_raw["PART_NAME"].to_numpy()
    restored["RH_LH"] = _extract_side_from_part_name(
        matched_raw["PART_NAME"]
    ).to_numpy()

    for col in ["PART_FACT_PLAN_DATE", "PART_FACT_SERIAL"]:
        if col in matched_raw.columns:
            restored[col] = matched_raw[col].to_numpy()

    if TARGET in restored.columns:
        target_scaled = _normalize_target(restored[TARGET])
        target_raw = _normalize_target(matched_raw[TARGET])

        target_match = (
            target_scaled.to_numpy()
            == target_raw.to_numpy()
        )

        match_rate = float(target_match.mean())

        print(f"Target match  : {match_rate:.4f}")

        if not target_match.all():
            mismatch_count = int((~target_match).sum())
            raise ValueError(
                "PassOrFail 복원이 일치하지 않습니다. "
                f"불일치 행={mismatch_count}"
            )

    side_counts = restored["RH_LH"].value_counts().to_dict()
    print(f"Side counts   : {side_counts}")

    return restored


def _prepare_date_counts(df: pd.DataFrame) -> dict:
    return (
        pd.to_datetime(df[TIME_COL])
        .dt.strftime("%Y-%m-%d")
        .value_counts()
        .sort_index()
        .to_dict()
    )


def preprocess_cn7(
    df: pd.DataFrame,
    require_target: bool = True,
    raw_labeled_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    최종 CN7 전처리.

    - TimeStamp가 없으면 labeled_data.csv에서 자동 복원
    - Side_RH 생성
    - PassOrFail: 정상=0 / 불량=1
    - 최종 모델 센서 23개 검증
    - 노트북과 동일하게 TimeStamp → RH_LH 순으로 정렬
    """
    work = df.copy()
    work.columns = work.columns.astype(str).str.strip()

    if TIME_COL not in work.columns:
        if raw_labeled_df is None:
            raise ValueError(
                "TimeStamp가 없으므로 labeled_data.csv가 필요합니다."
            )

        work = restore_cn7_metadata(
            scaled_df=work,
            raw_labeled_df=raw_labeled_df,
        )

    work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="coerce")
    if work[TIME_COL].isna().any():
        raise ValueError("TimeStamp datetime 변환에 실패한 행이 있습니다.")

    work[SIDE_FEATURE] = _make_side_rh(work)

    if require_target:
        if TARGET not in work.columns:
            raise ValueError(f"학습 데이터에 {TARGET}가 없습니다.")
        work[TARGET] = _normalize_target(work[TARGET])
    elif TARGET in work.columns:
        work[TARGET] = _normalize_target(work[TARGET])

    missing_sensors = [c for c in SENSOR_COLS if c not in work.columns]
    if missing_sensors:
        raise ValueError(f"필수 센서 컬럼 누락: {missing_sensors}")

    sensor_df = work[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")

    if sensor_df.isna().any().any():
        bad_cols = sensor_df.columns[sensor_df.isna().any()].tolist()
        raise ValueError(f"센서 숫자 변환/결측 문제가 있는 컬럼: {bad_cols}")

    if np.isinf(sensor_df.to_numpy(dtype=float)).any():
        raise ValueError("센서 데이터에 inf/-inf가 있습니다.")

    work[SENSOR_COLS] = sensor_df

    if "RH_LH" in work.columns:
        work = work.sort_values(
            [TIME_COL, "RH_LH"]
        ).reset_index(drop=True)
    else:
        # Side_RH: LH=0, RH=1이므로 동일한 순서
        work = work.sort_values(
            [TIME_COL, SIDE_FEATURE]
        ).reset_index(drop=True)

    keep_cols = [TIME_COL]

    if TARGET in work.columns:
        keep_cols.append(TARGET)

    keep_cols += [SIDE_FEATURE] + SENSOR_COLS

    result = work[keep_cols].copy()

    if len(result) == EXPECTED_TRAIN_ROWS and TARGET in result.columns:
        target_counts = result[TARGET].value_counts().sort_index().to_dict()
        if target_counts != {0: 1194, 1: 17}:
            raise ValueError(
                "CN7 최종 Target 분포가 예상과 다릅니다. "
                f"현재={target_counts}, 기대={{0: 1194, 1: 17}}"
            )

    return result


def preprocess_file(
    input_path,
    output_path,
    require_target: bool = True,
    raw_labeled_path=RAW_LABELED_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    raw_labeled_df = None

    if TIME_COL not in df.columns:
        if not Path(raw_labeled_path).exists():
            raise FileNotFoundError(
                "moldset_labeled_cn7.csv에 TimeStamp가 없는데 "
                f"복원용 원본도 없습니다: {raw_labeled_path}"
            )

        raw_labeled_df = pd.read_csv(raw_labeled_path)

    result = preprocess_cn7(
        df,
        require_target=require_target,
        raw_labeled_df=raw_labeled_df,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return result


if __name__ == "__main__":
    from .config_cn7 import (
        PREPROCESSED_PATH,
        RAW_MOLDSET_PATH,
    )

    result = preprocess_file(
        RAW_MOLDSET_PATH,
        PREPROCESSED_PATH,
        require_target=True,
        raw_labeled_path=RAW_LABELED_PATH,
    )

    print(f"\n저장 완료: {PREPROCESSED_PATH}")
    print("Shape:", result.shape)
