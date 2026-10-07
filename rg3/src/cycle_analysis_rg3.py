# ============================================================
# cycle_analysis_rg3.py
#
# RG3 추가 구조진단
# - Side-level PassOrFail → Cycle-level Target 재구성
# - 동일 TimeStamp에서 하나라도 불량이면 Cycle_Defect = 1
# - Session 내부 Absolute EWMA5 생성
# - 2020-10-21~10-23 학습
# - 2020-11-04 External Temporal Analysis
#
# 주의:
# 본 분석은 기존 RG3 최종모델을 대체하지 않음.
# 기존 미래 Session 실패 이후 수행한 사후 구조진단임.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

PROCESS_COLS = [
    "Injection_Time",
    "Filling_Time",
    "Plasticizing_Time",
    "Cycle_Time",
    "Clamp_Close_Time",
    "Cushion_Position",
    "Plasticizing_Position",
    "Max_Injection_Speed",
    "Max_Screw_RPM",
    "Average_Screw_RPM",
    "Max_Injection_Pressure",
    "Max_Switch_Over_Pressure",
    "Max_Back_Pressure",
    "Average_Back_Pressure",
    "Barrel_Temperature_1",
    "Barrel_Temperature_2",
    "Barrel_Temperature_3",
    "Barrel_Temperature_4",
    "Barrel_Temperature_5",
    "Barrel_Temperature_6",
    "Hopper_Temperature",
    "Mold_Temperature_3",
    "Mold_Temperature_4",
]

SESSION_GAP_MINUTES = 30
EWMA_SPAN = 5

TRAIN_END_DATE = pd.Timestamp("2020-10-23").date()
OOT_DATE = pd.Timestamp("2020-11-04").date()

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_rg3_raw(data_path: str | Path) -> pd.DataFrame:

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {data_path}"
        )

    df = pd.read_csv(data_path)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required = {
        "TimeStamp",
        "PART_NAME",
        "PassOrFail",
        *PROCESS_COLS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"필수 컬럼 누락: {sorted(missing)}"
        )

    df["TimeStamp"] = pd.to_datetime(
        df["TimeStamp"]
    )

    df = (
        df[
            df["PART_NAME"]
            .astype(str)
            .str.contains(
                "RG3",
                case=False,
                na=False,
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 2. TARGET / SIDE RESTORE
# ============================================================

def restore_target_and_side(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["Side"] = np.select(
        [
            df["PART_NAME"]
            .astype(str)
            .str.contains(
                "RH",
                case=False,
                na=False,
            ),

            df["PART_NAME"]
            .astype(str)
            .str.contains(
                "LH",
                case=False,
                na=False,
            ),
        ],
        [
            "RH",
            "LH",
        ],
        default="UNKNOWN",
    )

    # Y = 정상 / N = 불량
    if pd.api.types.is_numeric_dtype(
        df["PassOrFail"]
    ):
        df["Defect"] = (
            df["PassOrFail"]
            .astype(int)
        )

    else:
        df["Defect"] = (
            df["PassOrFail"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({
                "Y": 0,
                "N": 1,
            })
        )

    if df["Defect"].isna().any():
        bad_values = (
            df.loc[
                df["Defect"].isna(),
                "PassOrFail",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"PassOrFail 변환 실패: {bad_values}"
        )

    df["Defect"] = (
        df["Defect"]
        .astype(int)
    )

    return df


# ============================================================
# 3. SIDE SUMMARY
# ============================================================

def make_side_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    temp = df.copy()

    temp["Date"] = (
        temp["TimeStamp"]
        .dt.date
    )

    result = (
        temp
        .groupby(
            [
                "Date",
                "Side",
            ]
        )
        .agg(
            Rows=(
                "Defect",
                "size",
            ),
            Defects=(
                "Defect",
                "sum",
            ),
        )
        .reset_index()
    )

    return result


# ============================================================
# 4. SIDE → CYCLE
# ============================================================

def build_cycle_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    cycle_df = (
        df
        .groupby(
            "TimeStamp",
            as_index=False,
        )
        .agg({
            **{
                col: "median"
                for col in PROCESS_COLS
            },
            "Defect": "max",
        })
        .rename(
            columns={
                "Defect": "Cycle_Defect"
            }
        )
        .sort_values(
            "TimeStamp"
        )
        .reset_index(drop=True)
    )

    return cycle_df


# ============================================================
# 5. SESSION
# ============================================================

def add_session(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["Time_Diff"] = (
        df["TimeStamp"]
        .diff()
    )

    session_start = (
        df["Time_Diff"].isna()
        |
        (
            df["Time_Diff"]
            >=
            pd.Timedelta(
                minutes=SESSION_GAP_MINUTES
            )
        )
    )

    df["Session"] = (
        session_start
        .cumsum()
        .astype(int)
    )

    return df


# ============================================================
# 6. ABSOLUTE EWMA5
# ============================================================

def add_absolute_ewma(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:

    df = df.copy()

    feature_dict = {}
    feature_cols = []

    for col in PROCESS_COLS:

        past_ewma = (
            df
            .groupby(
                "Session",
                sort=False,
            )[col]
            .transform(
                lambda x:
                x.shift(1)
                .ewm(
                    span=EWMA_SPAN,
                    adjust=False,
                )
                .mean()
            )
        )

        new_col = (
            f"{col}_cycle_abs_ewma5"
        )

        feature_dict[new_col] = (
            df[col]
            -
            past_ewma
        ).abs()

        feature_cols.append(
            new_col
        )

    feature_df = pd.DataFrame(
        feature_dict,
        index=df.index,
    )

    df = pd.concat(
        [
            df,
            feature_df,
        ],
        axis=1,
    )

    return df, feature_cols


# ============================================================
# 7. SESSION SUMMARY
# ============================================================

def make_session_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df
        .groupby("Session")
        .agg(
            Start=(
                "TimeStamp",
                "min",
            ),
            End=(
                "TimeStamp",
                "max",
            ),
            Cycles=(
                "Cycle_Defect",
                "size",
            ),
            Defect_Cycles=(
                "Cycle_Defect",
                "sum",
            ),
        )
        .reset_index()
    )


# ============================================================
# 8. TRAIN / EXTERNAL OOT SPLIT
# ============================================================

def split_external_oot(
    df: pd.DataFrame,
    feature_cols: list[str],
):

    dates = (
        df["TimeStamp"]
        .dt.date
    )

    # 기존 개발구간까지만 Train
    train_df = (
        df[
            dates <= TRAIN_END_DATE
        ]
        .dropna(
            subset=feature_cols
        )
        .copy()
    )

    # 11/04만 추가 시간구간
    test_df = (
        df[
            dates == OOT_DATE
        ]
        .dropna(
            subset=feature_cols
        )
        .copy()
    )

    if train_df.empty:
        raise ValueError(
            "Train Cycle이 없습니다."
        )

    if test_df.empty:
        raise ValueError(
            "2020-11-04 OOT Cycle이 없습니다."
        )

    return train_df, test_df


# ============================================================
# 9. MODEL
# ============================================================

def train_cycle_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
):

    X_train = (
        train_df[
            feature_cols
        ]
    )

    y_train = (
        train_df[
            "Cycle_Defect"
        ]
        .astype(int)
    )

    neg = int(
        (y_train == 0).sum()
    )

    pos = int(
        (y_train == 1).sum()
    )

    if pos == 0:
        raise ValueError(
            "Train 데이터에 불량 Cycle이 없습니다."
        )

    scale_pos_weight = (
        neg / pos
    )

    model = XGBClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# 10. PERFORMANCE
# ============================================================

def evaluate_external_oot(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
):

    X_test = (
        test_df[
            feature_cols
        ]
    )

    y_test = (
        test_df[
            "Cycle_Defect"
        ]
        .astype(int)
    )

    risk_score = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    pr_auc = (
        average_precision_score(
            y_test,
            risk_score,
        )
    )

    roc_auc = (
        roc_auc_score(
            y_test,
            risk_score,
        )
    )

    prevalence = (
        y_test.mean()
    )

    pr_lift = (
        pr_auc / prevalence
        if prevalence > 0
        else np.nan
    )

    prediction_df = (
        test_df[
            [
                "TimeStamp",
                "Session",
                "Cycle_Defect",
            ]
        ]
        .copy()
    )

    prediction_df[
        "Risk_Score"
    ] = risk_score

    prediction_df[
        "Risk_Rank"
    ] = (
        prediction_df[
            "Risk_Score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    metrics = {
        "Test_Cycles":
            len(test_df),

        "Defect_Cycles":
            int(
                y_test.sum()
            ),

        "Prevalence":
            prevalence,

        "PR_AUC":
            pr_auc,

        "ROC_AUC":
            roc_auc,

        "PR_Lift":
            pr_lift,
    }

    return (
        metrics,
        prediction_df,
    )


# ============================================================
# 11. INSPECTION TRADE-OFF
# ============================================================

def make_inspection_tradeoff(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    total = len(
        prediction_df
    )

    total_defects = int(
        prediction_df[
            "Cycle_Defect"
        ]
        .sum()
    )

    for rate in [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]:

        inspect_n = max(
            1,
            int(
                np.ceil(
                    total * rate
                )
            ),
        )

        selected = (
            prediction_df
            .nsmallest(
                inspect_n,
                "Risk_Rank",
            )
        )

        tp = int(
            selected[
                "Cycle_Defect"
            ]
            .sum()
        )

        fn = (
            total_defects
            -
            tp
        )

        precision = (
            tp / inspect_n
            if inspect_n > 0
            else 0.0
        )

        recall = (
            tp / total_defects
            if total_defects > 0
            else 0.0
        )

        rows.append({
            "Inspection_Rate":
                rate,

            "Inspection_Count":
                inspect_n,

            "TP":
                tp,

            "FN":
                fn,

            "Precision":
                precision,

            "Recall":
                recall,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. MAIN ANALYSIS
# ============================================================

def run_cycle_analysis(
    data_path: str | Path,
    output_dir: str | Path = "outputs",
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Raw
    # --------------------------------------------------------

    raw = load_rg3_raw(
        data_path
    )

    raw = restore_target_and_side(
        raw
    )

    side_summary = make_side_summary(
        raw
    )

    # --------------------------------------------------------
    # Cycle
    # --------------------------------------------------------

    cycle_df = build_cycle_data(
        raw
    )

    cycle_df = add_session(
        cycle_df
    )

    cycle_df, feature_cols = (
        add_absolute_ewma(
            cycle_df
        )
    )

    session_summary = (
        make_session_summary(
            cycle_df
        )
    )

    # --------------------------------------------------------
    # Train / External OOT
    # --------------------------------------------------------

    train_df, test_df = (
        split_external_oot(
            cycle_df,
            feature_cols,
        )
    )

    model = train_cycle_model(
        train_df,
        feature_cols,
    )

    metrics, predictions = (
        evaluate_external_oot(
            model,
            test_df,
            feature_cols,
        )
    )

    tradeoff = (
        make_inspection_tradeoff(
            predictions
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    side_summary.to_csv(
        output_dir
        /
        "rg3_cycle_side_summary.csv",
        index=False,
    )

    session_summary.to_csv(
        output_dir
        /
        "rg3_cycle_session_summary.csv",
        index=False,
    )

    predictions.to_csv(
        output_dir
        /
        "rg3_cycle_oot_predictions.csv",
        index=False,
    )

    tradeoff.to_csv(
        output_dir
        /
        "rg3_cycle_inspection_tradeoff.csv",
        index=False,
    )

    metrics_df = pd.DataFrame([
        metrics
    ])

    metrics_df.to_csv(
        output_dir
        /
        "rg3_cycle_oot_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "=" * 100
    )

    print(
        "RG3 CYCLE-LEVEL ADDITIONAL ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        "\n[DATE × SIDE DEFECT]"
    )

    print(
        side_summary.to_string(
            index=False
        )
    )

    print(
        "\n[SESSION STRUCTURE]"
    )

    print(
        session_summary.to_string(
            index=False
        )
    )

    print(
        "\n[EXTERNAL OOT]"
    )

    print(
        f"Train : "
        f"{train_df['TimeStamp'].min()} "
        f"~ "
        f"{train_df['TimeStamp'].max()}"
    )

    print(
        f"Train Cycles  : "
        f"{len(train_df)}"
    )

    print(
        f"Train Defects : "
        f"{int(train_df['Cycle_Defect'].sum())}"
    )

    print()

    print(
        f"Test : "
        f"{test_df['TimeStamp'].min()} "
        f"~ "
        f"{test_df['TimeStamp'].max()}"
    )

    print(
        f"Test Cycles   : "
        f"{metrics['Test_Cycles']}"
    )

    print(
        f"Test Defects  : "
        f"{metrics['Defect_Cycles']}"
    )

    print(
        f"Prevalence    : "
        f"{metrics['Prevalence']:.4f}"
    )

    print(
        f"PR-AUC        : "
        f"{metrics['PR_AUC']:.4f}"
    )

    print(
        f"ROC-AUC       : "
        f"{metrics['ROC_AUC']:.4f}"
    )

    print(
        f"PR Lift       : "
        f"{metrics['PR_Lift']:.3f}x"
    )

    print(
        "\n[INSPECTION TRADE-OFF]"
    )

    print(
        tradeoff
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n[DEFECT CYCLE RANK]"
    )

    print(
        predictions[
            predictions[
                "Cycle_Defect"
            ]
            ==
            1
        ]
        .sort_values(
            "Risk_Rank"
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n저장 완료:"
    )

    print(
        output_dir
        /
        "rg3_cycle_side_summary.csv"
    )

    print(
        output_dir
        /
        "rg3_cycle_session_summary.csv"
    )

    print(
        output_dir
        /
        "rg3_cycle_oot_metrics.csv"
    )

    print(
        output_dir
        /
        "rg3_cycle_oot_predictions.csv"
    )

    print(
        output_dir
        /
        "rg3_cycle_inspection_tradeoff.csv"
    )

    return {
        "metrics":
            metrics_df,

        "predictions":
            predictions,

        "tradeoff":
            tradeoff,

        "side_summary":
            side_summary,

        "session_summary":
            session_summary,

        "model":
            model,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]
    DATA_PATH = project_root.parent / "data" / "raw" / "labeled_data.csv"

    run_cycle_analysis(
        data_path=DATA_PATH,
        output_dir=project_root / "outputs",
    )
