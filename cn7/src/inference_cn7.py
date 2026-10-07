import joblib
import numpy as np
import pandas as pd

from .config_cn7 import (
    CV_SEEDS,
    FEATURE_COLS,
    HIGH_PRECISION_THRESHOLD,
    HIGH_RECALL_THRESHOLD,
    INFERENCE_PATH,
    MODEL_DIR,
    MODEL_INPUT_PATH,
    OPERATION_THRESHOLD,
    OUTPUT_DIR,
    SIDE_FEATURE,
    TARGET,
    TIME_COL,
)


def _load_models(model_dir=MODEL_DIR):
    artifacts = []

    for seed in CV_SEEDS:
        path = model_dir / f"cn7_lgbm_seed{seed}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
        artifacts.append(joblib.load(path))

    return artifacts


def _risk_level(prob: float) -> str:
    if prob >= HIGH_PRECISION_THRESHOLD:
        return "CRITICAL"
    if prob >= OPERATION_THRESHOLD:
        return "HIGH"
    if prob >= HIGH_RECALL_THRESHOLD:
        return "WATCH"
    return "LOW"


def _priority(row) -> str:
    if row["Risk_Probability"] >= HIGH_PRECISION_THRESHOLD:
        return "P1_CRITICAL"
    if row["Risk_Probability"] >= OPERATION_THRESHOLD:
        return "P2_HIGH"
    if row["Prediction_Stability"] == "Unstable":
        return "P3_UNCERTAIN"
    return "P4_NORMAL"


def inference_cn7(
    model_input_path=MODEL_INPUT_PATH,
    output_path=INFERENCE_PATH,
    model_dir=MODEL_DIR,
) -> pd.DataFrame:
    df = pd.read_csv(model_input_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"추론 Feature 누락: {missing}")

    artifacts = _load_models(model_dir)
    X = df[FEATURE_COLS].copy()

    model_proba = []
    for artifact in artifacts:
        model = artifact["model"]
        model_proba.append(model.predict_proba(X)[:, 1])

    matrix = np.column_stack(model_proba)
    risk = matrix.mean(axis=1)
    risk_std = matrix.std(axis=1)
    votes = (matrix >= OPERATION_THRESHOLD).sum(axis=1)

    result_cols = [c for c in [TIME_COL, SIDE_FEATURE, TARGET] if c in df.columns]
    result = df[result_cols].copy()

    for seed, proba in zip(CV_SEEDS, model_proba):
        result[f"Proba_Model_Seed_{seed}"] = proba

    result["Risk_Probability"] = risk
    # 표시용 0~100 점수. 운영 threshold 0.060은 Risk_Probability 기준이며
    # Risk_Score 기준으로는 6.0과 동일하다.
    result["Risk_Score"] = risk * 100
    result["Risk_Threshold_Probability"] = OPERATION_THRESHOLD
    result["Risk_Threshold_Score"] = OPERATION_THRESHOLD * 100
    result["Risk_STD"] = risk_std
    result["Positive_Votes"] = votes
    result["Prediction_Stability"] = np.where(
        (votes == 0) | (votes == len(CV_SEEDS)),
        "Stable",
        "Unstable",
    )
    result["Risk_Level"] = [_risk_level(p) for p in risk]
    result["Risk_Only_Flag"] = (risk >= OPERATION_THRESHOLD).astype(int)
    result["Final_Inspection_Flag"] = (
        (result["Risk_Only_Flag"] == 1)
        | (result["Prediction_Stability"] == "Unstable")
    ).astype(int)

    result["Inspection_Priority"] = result.apply(_priority, axis=1)

    priority_order = {
        "P1_CRITICAL": 1,
        "P2_HIGH": 2,
        "P3_UNCERTAIN": 3,
        "P4_NORMAL": 4,
    }
    action_map = {
        "P1_CRITICAL": "즉시 검사",
        "P2_HIGH": "우선 검사",
        "P3_UNCERTAIN": "불확실성 추가 검사",
        "P4_NORMAL": "일반 검사",
    }

    result["_Priority_Order"] = result["Inspection_Priority"].map(priority_order)
    result["Inspection_Action"] = result["Inspection_Priority"].map(action_map)

    result = result.sort_values(
        ["_Priority_Order", "Risk_Probability", "Risk_STD"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    result["Inspection_Rank"] = np.arange(1, len(result) + 1)
    result = result.drop(columns="_Priority_Order")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"추론 결과 저장: {output_path}")
    return result


if __name__ == "__main__":
    inference_cn7()
