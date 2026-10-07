import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from .config_cn7 import (
    CV_SEEDS,
    METRICS_PATH,
    OOF_PATH,
    OPERATION_THRESHOLD,
    OUTPUT_DIR,
    TARGET,
    THRESHOLD_PATH,
)


def _metrics(y_true, pred, proba=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result = {
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "F2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Flag_Count": int(pred.sum()),
        "Flag_Rate_pct": float(pred.mean() * 100),
    }
    if proba is not None:
        result["PR_AUC"] = average_precision_score(y_true, proba)
    return result


def threshold_search(oof_df: pd.DataFrame) -> pd.DataFrame:
    y = oof_df[TARGET].astype(int).to_numpy()
    proba_cols = [f"Proba_Seed_{s}" for s in CV_SEEDS]

    rows = []
    for threshold in np.arange(0.005, 0.501, 0.005):
        seed_rows = []

        for col in proba_cols:
            proba = oof_df[col].to_numpy()
            pred = (proba >= threshold).astype(int)
            seed_rows.append(_metrics(y, pred, proba))

        temp = pd.DataFrame(seed_rows)

        row = {"Threshold": round(float(threshold), 3)}
        for metric in [
            "Precision",
            "Recall",
            "F1",
            "F2",
            "TP",
            "FP",
            "FN",
            "Flag_Rate_pct",
        ]:
            row[f"{metric}_mean"] = temp[metric].mean()
            row[f"{metric}_std"] = temp[metric].std(ddof=0)

        rows.append(row)

    return pd.DataFrame(rows)


def build_metrics(oof_df: pd.DataFrame) -> pd.DataFrame:
    y = oof_df[TARGET].astype(int).to_numpy()
    rows = []

    # 1) seed별 OOF 성능: 노트북의 반복 SGKF 성능
    for seed in CV_SEEDS:
        col = f"Proba_Seed_{seed}"
        proba = oof_df[col].to_numpy()
        pred = (proba >= OPERATION_THRESHOLD).astype(int)

        row = {
            "Scope": "PerSeed_OOF",
            "Policy": f"Seed_{seed}",
            "Threshold": OPERATION_THRESHOLD,
        }
        row.update(_metrics(y, pred, proba))
        rows.append(row)

    per_seed = pd.DataFrame(rows)
    numeric_cols = [
        "PR_AUC",
        "Precision",
        "Recall",
        "F1",
        "F2",
        "TP",
        "FP",
        "FN",
        "TN",
        "Flag_Count",
        "Flag_Rate_pct",
    ]

    mean_row = {
        "Scope": "PerSeed_Mean",
        "Policy": "Mean_of_5_CV_Seeds",
        "Threshold": OPERATION_THRESHOLD,
    }
    std_row = {
        "Scope": "PerSeed_Std",
        "Policy": "Std_of_5_CV_Seeds",
        "Threshold": OPERATION_THRESHOLD,
    }
    for col in numeric_cols:
        mean_row[col] = per_seed[col].mean()
        std_row[col] = per_seed[col].std(ddof=0)

    rows.extend([mean_row, std_row])

    # 2) 5개 OOF 확률 평균 기반 Risk-only / Risk+Uncertainty 정책
    proba_cols = [f"Proba_Seed_{s}" for s in CV_SEEDS]
    matrix = oof_df[proba_cols].to_numpy()
    risk = matrix.mean(axis=1)
    votes = (matrix >= OPERATION_THRESHOLD).sum(axis=1)
    unstable = (votes > 0) & (votes < len(CV_SEEDS))

    risk_only = (risk >= OPERATION_THRESHOLD).astype(int)
    risk_uncertainty = ((risk >= OPERATION_THRESHOLD) | unstable).astype(int)

    for name, pred in [
        ("Risk_Only", risk_only),
        ("Risk_Plus_Uncertainty", risk_uncertainty),
    ]:
        row = {
            "Scope": "Ensemble_OOF",
            "Policy": name,
            "Threshold": OPERATION_THRESHOLD,
        }
        row.update(_metrics(y, pred, risk))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_cn7(
    oof_path=OOF_PATH,
    threshold_path=THRESHOLD_PATH,
    metrics_path=METRICS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_df = pd.read_csv(oof_path)

    threshold_df = threshold_search(oof_df)
    metrics_df = build_metrics(oof_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"Threshold 결과 저장: {threshold_path}")
    print(f"Metrics 저장: {metrics_path}")

    mean_row = metrics_df[metrics_df["Scope"] == "PerSeed_Mean"].iloc[0]
    print(
        f"Threshold {OPERATION_THRESHOLD:.3f} | "
        f"Precision {mean_row['Precision']:.4f} | "
        f"Recall {mean_row['Recall']:.4f} | "
        f"F1 {mean_row['F1']:.4f} | "
        f"F2 {mean_row['F2']:.4f}"
    )

    return threshold_df, metrics_df


if __name__ == "__main__":
    evaluate_cn7()
