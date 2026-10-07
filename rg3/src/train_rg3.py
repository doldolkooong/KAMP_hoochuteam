import argparse
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

try:
    from src.config_rg3 import (
        MODEL_FEATURES,
        MODEL_INPUT_PATH,
        MODEL_PARAMS,
        MODEL_PATH,
        TARGET,
    )
except ModuleNotFoundError:
    from config_rg3 import (
        MODEL_FEATURES,
        MODEL_INPUT_PATH,
        MODEL_PARAMS,
        MODEL_PATH,
        TARGET,
    )


def build_xgb_model(y_train: pd.Series) -> XGBClassifier:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())

    if pos == 0:
        raise ValueError("Train 데이터에 불량(1)이 없습니다.")

    return XGBClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=neg / pos,
    )


def train_final_model(
    input_path: Path = MODEL_INPUT_PATH,
    model_path: Path = MODEL_PATH,
):
    input_path = Path(input_path)
    model_path = Path(model_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"모델 입력 데이터가 없습니다: {input_path}"
        )

    df = pd.read_csv(input_path)

    missing = [
        c for c in MODEL_FEATURES + [TARGET]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"학습에 필요한 컬럼이 없습니다: {missing}")

    X = df[MODEL_FEATURES]
    y = df[TARGET].astype(int)

    model = build_xgb_model(y)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_names": MODEL_FEATURES,
        "target": TARGET,
        "model_params": MODEL_PARAMS,
        "train_rows": len(df),
        "normal_count": int((y == 0).sum()),
        "defect_count": int((y == 1).sum()),
        "scale_pos_weight": float((y == 0).sum() / (y == 1).sum()),
        "usage_note": (
            "RG3는 미래 Session별 성능 변동이 확인된 Risk Ranking 보조 모델입니다. "
            "낮은 Risk Score를 자동 정상판정으로 사용하지 마세요."
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    print("=" * 72)
    print("RG3 FINAL MODEL SAVED")
    print("=" * 72)
    print(f"Rows             : {len(df)}")
    print(f"Normal           : {bundle['normal_count']}")
    print(f"Defect           : {bundle['defect_count']}")
    print(f"scale_pos_weight : {bundle['scale_pos_weight']:.4f}")
    print(f"Features         : {len(MODEL_FEATURES)}")
    print(f"Saved            : {model_path}")

    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=MODEL_INPUT_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    train_final_model(args.input, args.model)


if __name__ == "__main__":
    main()
