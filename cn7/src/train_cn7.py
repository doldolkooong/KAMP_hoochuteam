import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from .config_cn7 import (
    BEST_PARAMS,
    CLASS_WEIGHT,
    CV_MODEL_RANDOM_STATE,
    CV_SEEDS,
    FEATURE_COLS,
    F1_THRESHOLD,
    MODEL_DIR,
    MODEL_INPUT_PATH,
    N_SPLITS,
    OOF_PATH,
    OPERATION_THRESHOLD,
    OUTPUT_DIR,
    SIDE_FEATURE,
    TARGET,
    TIME_COL,
)


def make_model(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        **BEST_PARAMS,
        class_weight=CLASS_WEIGHT,
        random_state=random_state,
        verbose=-1,
    )


def repeated_group_oof(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = model_df[FEATURE_COLS].copy()
    y = model_df[TARGET].astype(int).copy()
    groups = pd.to_datetime(model_df[TIME_COL]).copy()

    oof_df = model_df[[TIME_COL, SIDE_FEATURE, TARGET]].copy()
    metrics = []

    for split_seed in CV_SEEDS:
        sgkf = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=split_seed,
        )

        oof_proba = np.zeros(len(model_df), dtype=float)

        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(X, y, groups=groups),
            start=1,
        ):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            # 노트북 재현: split seed만 바꾸고 모델 random_state는 42로 고정
            model = make_model(CV_MODEL_RANDOM_STATE)
            model.fit(X_train, y_train)
            oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]

        oof_df[f"Proba_Seed_{split_seed}"] = oof_proba
        pr_auc = average_precision_score(y, oof_proba)
        metrics.append({"Seed": split_seed, "PR_AUC": pr_auc})

    proba_cols = [f"Proba_Seed_{s}" for s in CV_SEEDS]
    matrix = oof_df[proba_cols].to_numpy()

    oof_df["Risk_Probability"] = matrix.mean(axis=1)
    oof_df["Risk_STD"] = matrix.std(axis=1)
    oof_df["Positive_Votes"] = (matrix >= OPERATION_THRESHOLD).sum(axis=1)
    oof_df["Prediction_Stability"] = np.where(
        (oof_df["Positive_Votes"] == 0)
        | (oof_df["Positive_Votes"] == len(CV_SEEDS)),
        "Stable",
        "Unstable",
    )

    return oof_df, pd.DataFrame(metrics)


def train_full_models(model_df: pd.DataFrame) -> list:
    """
    전체 학습 데이터로 5개 운영용 ensemble 모델을 학습한다.

    OOF 재현 단계에서는 모델 seed=42를 고정하지만,
    운영 ensemble은 5개 파일이 실제로 서로 다른 모델이 되도록
    random_state를 각 seed로 설정한다.
    """
    X = model_df[FEATURE_COLS].copy()
    y = model_df[TARGET].astype(int).copy()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for model_seed in CV_SEEDS:
        model = make_model(model_seed)
        model.fit(X, y)

        artifact = {
            "model": model,
            "features": FEATURE_COLS,
            "model_seed": model_seed,
            "threshold_f1": F1_THRESHOLD,
            "threshold_operation": OPERATION_THRESHOLD,
            "best_params": BEST_PARAMS,
            "class_weight": CLASS_WEIGHT,
        }

        path = MODEL_DIR / f"cn7_lgbm_seed{model_seed}.pkl"
        joblib.dump(artifact, path)
        paths.append(path)

    return paths


def train_cn7(
    model_input_path=MODEL_INPUT_PATH,
    oof_path=OOF_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = pd.read_csv(model_input_path)

    missing = [c for c in [TIME_COL, TARGET] + FEATURE_COLS if c not in model_df.columns]
    if missing:
        raise ValueError(f"모델 입력 컬럼 누락: {missing}")

    oof_df, seed_pr_auc_df = repeated_group_oof(model_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    model_paths = train_full_models(model_df)

    print(f"OOF 저장: {oof_path}")
    for path in model_paths:
        print(f"모델 저장: {path}")

    print(
        "Repeated SGKF PR-AUC:",
        f"{seed_pr_auc_df['PR_AUC'].mean():.4f}",
        "±",
        f"{seed_pr_auc_df['PR_AUC'].std(ddof=0):.4f}",
    )

    return oof_df, seed_pr_auc_df


if __name__ == "__main__":
    train_cn7()
