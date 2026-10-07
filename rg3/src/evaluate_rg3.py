import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from src.config_rg3 import (
        HOLDOUT_PREDICTION_PATH,
        MODEL_FEATURES,
        MODEL_INPUT_PATH,
        ORIGINAL_INDEX_COL,
        TARGET,
        TEMPORAL_RESULT_PATH,
        TIMESTAMP_COL,
    )
    from src.train_rg3 import build_xgb_model
except ModuleNotFoundError:
    from config_rg3 import (
        HOLDOUT_PREDICTION_PATH,
        MODEL_FEATURES,
        MODEL_INPUT_PATH,
        ORIGINAL_INDEX_COL,
        TARGET,
        TEMPORAL_RESULT_PATH,
        TIMESTAMP_COL,
    )
    from train_rg3 import build_xgb_model


TEMPORAL_FOLDS = [
    {"Fold": 1, "Train_Sessions": [1, 2], "Valid_Session": 3},
    {"Fold": 2, "Train_Sessions": [1, 2, 3], "Valid_Session": 4},
    {"Fold": 3, "Train_Sessions": [1, 2, 3, 4], "Valid_Session": 5},
]


def safe_roc_auc(y_true, score):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, score)


def evaluate_scores(y_true, score):
    y_true = pd.Series(y_true).astype(int)
    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, score))
    roc_auc = float(safe_roc_auc(y_true, score))
    pr_lift = (
        float(pr_auc / prevalence)
        if prevalence > 0
        else np.nan
    )
    return {
        "Defect_Rate": prevalence,
        "PR_AUC": pr_auc,
        "PR_Lift": pr_lift,
        "ROC_AUC": roc_auc,
    }


def rolling_temporal_validation(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    available_sessions = set(df["Session"].astype(int).unique())

    for fold in TEMPORAL_FOLDS:
        train_sessions = fold["Train_Sessions"]
        valid_session = fold["Valid_Session"]

        needed = set(train_sessions + [valid_session])
        if not needed.issubset(available_sessions):
            print(
                f"[WARN] Fold {fold['Fold']} 건너뜀: "
                f"필요 Session {sorted(needed)}"
            )
            continue

        train = df[df["Session"].isin(train_sessions)].copy()
        valid = df[df["Session"] == valid_session].copy()

        X_train = train[MODEL_FEATURES]
        y_train = train[TARGET].astype(int)
        X_valid = valid[MODEL_FEATURES]
        y_valid = valid[TARGET].astype(int)

        model = build_xgb_model(y_train)
        model.fit(X_train, y_train)
        score = model.predict_proba(X_valid)[:, 1]

        metrics = evaluate_scores(y_valid, score)

        results.append(
            {
                "Fold": fold["Fold"],
                "Train_Sessions": ",".join(map(str, train_sessions)),
                "Valid_Session": valid_session,
                "Valid_Count": len(valid),
                "Defect_Count": int(y_valid.sum()),
                **metrics,
            }
        )

    return pd.DataFrame(results)


def final_holdout_evaluation(df: pd.DataFrame):
    train = df[df["Session"].isin([1, 2, 3])].copy()
    holdout = df[df["Session"].isin([4, 5])].copy()

    if train.empty or holdout.empty:
        raise ValueError(
            "Final Holdout 평가에는 Session 1~5가 필요합니다."
        )

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET].astype(int)
    X_holdout = holdout[MODEL_FEATURES]
    y_holdout = holdout[TARGET].astype(int)

    model = build_xgb_model(y_train)
    model.fit(X_train, y_train)

    score = model.predict_proba(X_holdout)[:, 1]
    metrics = evaluate_scores(y_holdout, score)

    keep_cols = [
        c for c in [
            ORIGINAL_INDEX_COL,
            TIMESTAMP_COL,
            "Session",
            TARGET,
        ]
        if c in holdout.columns
    ]

    pred = holdout[keep_cols].copy()
    pred["Risk_Score"] = score
    pred["Risk_Rank"] = (
        pred["Risk_Score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return metrics, pred


def run_evaluation(
    input_path: Path = MODEL_INPUT_PATH,
    temporal_output: Path = TEMPORAL_RESULT_PATH,
    holdout_output: Path = HOLDOUT_PREDICTION_PATH,
):
    input_path = Path(input_path)
    temporal_output = Path(temporal_output)
    holdout_output = Path(holdout_output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"모델 입력 데이터가 없습니다: {input_path}"
        )

    df = pd.read_csv(input_path)

    temporal = rolling_temporal_validation(df)
    temporal_output.parent.mkdir(parents=True, exist_ok=True)
    temporal.to_csv(temporal_output, index=False)

    holdout_metrics, holdout_pred = final_holdout_evaluation(df)
    holdout_pred.to_csv(holdout_output, index=False)

    print("=" * 72)
    print("RG3 TEMPORAL VALIDATION")
    print("=" * 72)
    if not temporal.empty:
        print(
            temporal[
                [
                    "Fold",
                    "Train_Sessions",
                    "Valid_Session",
                    "Valid_Count",
                    "Defect_Count",
                    "PR_AUC",
                    "PR_Lift",
                    "ROC_AUC",
                ]
            ].to_string(index=False)
        )

    print("\nFINAL HOLDOUT: Session 1~3 -> Session 4~5")
    for key, value in holdout_metrics.items():
        print(f"{key:12s}: {value:.6f}")

    print(f"\nSaved temporal : {temporal_output}")
    print(f"Saved holdout  : {holdout_output}")

    return temporal, holdout_metrics, holdout_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=MODEL_INPUT_PATH)
    parser.add_argument(
        "--temporal-output",
        type=Path,
        default=TEMPORAL_RESULT_PATH,
    )
    parser.add_argument(
        "--holdout-output",
        type=Path,
        default=HOLDOUT_PREDICTION_PATH,
    )
    args = parser.parse_args()

    run_evaluation(
        args.input,
        args.temporal_output,
        args.holdout_output,
    )


if __name__ == "__main__":
    main()
