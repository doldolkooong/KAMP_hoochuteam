import argparse
from pathlib import Path

import pandas as pd

try:
    from src.config_rg3 import (
        EWMA_SPAN,
        MODEL_FEATURES,
        MODEL_INPUT_PATH,
        PREPROCESSED_PATH,
        SENSOR_COLS,
        TARGET,
    )
except ModuleNotFoundError:
    from config_rg3 import (
        EWMA_SPAN,
        MODEL_FEATURES,
        MODEL_INPUT_PATH,
        PREPROCESSED_PATH,
        SENSOR_COLS,
        TARGET,
    )


def add_absolute_ewma_features(
    df: pd.DataFrame,
    sensor_cols=SENSOR_COLS,
    span: int = EWMA_SPAN,
    session_col: str = "Session",
) -> pd.DataFrame:
    out = df.copy()

    if session_col not in out.columns:
        raise KeyError(f"'{session_col}' 컬럼이 없습니다.")

    missing = [c for c in sensor_cols if c not in out.columns]
    if missing:
        raise KeyError(f"EWMA 생성에 필요한 센서가 없습니다: {missing}")

    for col in sensor_cols:
        past_ewma = (
            out.groupby(session_col, sort=False)[col]
            .transform(
                lambda x: (
                    x.shift(1)
                    .ewm(span=span, adjust=False)
                    .mean()
                )
            )
        )

        signed_name = f"{col}_ewma{span}_dev"
        abs_name = f"{col}_ewma{span}_abs_dev"

        out[signed_name] = out[col] - past_ewma
        out[abs_name] = out[signed_name].abs()

    return out


def build_model_input(
    input_path: Path = PREPROCESSED_PATH,
    output_path: Path = MODEL_INPUT_PATH,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"전처리 데이터가 없습니다: {input_path}"
        )

    df = pd.read_csv(input_path)
    engineered = add_absolute_ewma_features(df)

    missing_features = [
        c for c in MODEL_FEATURES if c not in engineered.columns
    ]
    if missing_features:
        raise RuntimeError(
            f"최종 Feature 생성 실패: {missing_features}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(output_path, index=False)

    print("=" * 72)
    print("RG3 FEATURE ENGINEERING COMPLETE")
    print("=" * 72)
    print(f"Absolute EWMA span : {EWMA_SPAN}")
    print(f"Final feature count: {len(MODEL_FEATURES)}")
    print(f"Saved              : {output_path}")

    return engineered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PREPROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_INPUT_PATH)
    args = parser.parse_args()

    build_model_input(args.input, args.output)


if __name__ == "__main__":
    main()
