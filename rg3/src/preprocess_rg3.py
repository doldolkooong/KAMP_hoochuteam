import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config_rg3 import (
        CONSTANT_COLS,
        LABELED_DATA_PATH,
        MOLDSET_PATH,
        ORIGINAL_INDEX_COL,
        PART_COL,
        PREPROCESSED_PATH,
        RAW_SENSOR_COLS,
        SENSOR_COLS,
        SESSION_GAP_MINUTES,
        SIDE_COL,
        SOURCE_INDEX_COL,
        TARGET,
        TIMESTAMP_COL,
    )
except ModuleNotFoundError:
    from config_rg3 import (
        CONSTANT_COLS,
        LABELED_DATA_PATH,
        MOLDSET_PATH,
        ORIGINAL_INDEX_COL,
        PART_COL,
        PREPROCESSED_PATH,
        RAW_SENSOR_COLS,
        SENSOR_COLS,
        SESSION_GAP_MINUTES,
        SIDE_COL,
        SOURCE_INDEX_COL,
        TARGET,
        TIMESTAMP_COL,
    )


def derive_side(part_name: pd.Series) -> pd.Series:
    part = part_name.astype(str)
    return pd.Series(
        np.select(
            [
                part.str.contains("RH", case=False, na=False),
                part.str.contains("LH", case=False, na=False),
            ],
            ["RH", "LH"],
            default="UNKNOWN",
        ),
        index=part_name.index,
        name=SIDE_COL,
    )


def restore_metadata(
    moldset: pd.DataFrame,
    labeled_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    moldset_labeled_rg3의 `Unnamed: 0` 값을 labeled_data.csv의
    0-based 원본 행 위치로 사용해 생산 메타정보를 복원한다.

    중요:
    - 모델 입력 센서값은 moldset_labeled_rg3의 표준화 값을 유지한다.
    - labeled_data.csv에서는 original_index / TimeStamp / PART_NAME만 복원한다.
    """
    df = moldset.copy()

    # 이미 복원된 파일도 처리 가능
    already_restored = (
        TIMESTAMP_COL in df.columns
        and PART_COL in df.columns
    )
    if already_restored:
        if ORIGINAL_INDEX_COL not in df.columns:
            if SOURCE_INDEX_COL in df.columns:
                df[ORIGINAL_INDEX_COL] = pd.to_numeric(
                    df[SOURCE_INDEX_COL], errors="raise"
                ).astype(int)
            else:
                df[ORIGINAL_INDEX_COL] = np.arange(len(df))
        if SIDE_COL not in df.columns:
            df[SIDE_COL] = derive_side(df[PART_COL])
        return df

    if SOURCE_INDEX_COL not in df.columns:
        raise KeyError(
            f"'{SOURCE_INDEX_COL}' 컬럼이 없어 labeled_data.csv 원본 행을 "
            "복원할 수 없습니다."
        )

    source_index = pd.to_numeric(
        df[SOURCE_INDEX_COL], errors="raise"
    ).astype(int)

    if source_index.isna().any():
        raise ValueError("원본 행 인덱스에 결측치가 있습니다.")

    if source_index.duplicated().any():
        raise ValueError(
            "moldset_labeled_rg3의 원본 행 인덱스가 중복됩니다. "
            "복원 구조를 다시 확인하세요."
        )

    if source_index.min() < 0 or source_index.max() >= len(labeled_raw):
        raise IndexError(
            "moldset의 원본 행 인덱스가 labeled_data.csv 범위를 벗어납니다."
        )

    raw_selected = (
        labeled_raw.iloc[source_index.to_numpy()]
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": ORIGINAL_INDEX_COL})
    )

    required_meta = [TIMESTAMP_COL, PART_COL]
    missing_meta = [c for c in required_meta if c not in raw_selected.columns]
    if missing_meta:
        raise KeyError(
            f"labeled_data.csv에 필요한 메타 컬럼이 없습니다: {missing_meta}"
        )

    # Target이 양쪽 모두 존재하면 행 복원이 맞는지 검증
    # moldset: 0=정상, 1=불량
    # labeled_data.csv: Y=정상(Pass), N=불량(Fail)
    if TARGET in df.columns and TARGET in raw_selected.columns:
        left = pd.to_numeric(
            df[TARGET],
            errors="coerce",
        ).reset_index(drop=True)

        raw_target = (
            raw_selected[TARGET]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        target_map = {
            "Y": 0,
            "N": 1,
            "0": 0,
            "1": 1,
            "0.0": 0,
            "1.0": 1,
        }

        right = (
            raw_target
            .map(target_map)
            .reset_index(drop=True)
        )

        unknown = right.isna()
        if unknown.any():
            unknown_values = sorted(
                raw_target.loc[unknown].unique().tolist()
            )
            raise ValueError(
                "labeled_data.csv의 PassOrFail 값 중 "
                f"해석할 수 없는 값이 있습니다: {unknown_values}"
            )

        target_match = (left == right).mean()

        print(
            f"Target match   : {target_match:.4f}"
        )

        if target_match < 1.0:
            mismatch = pd.DataFrame({
                "source_index": source_index.reset_index(drop=True),
                "moldset_target": left,
                "raw_target": raw_target,
                "raw_target_mapped": right,
            })

            mismatch = mismatch.loc[
                mismatch["moldset_target"]
                != mismatch["raw_target_mapped"]
            ]

            raise ValueError(
                "Target 매칭률이 100%가 아닙니다: "
                f"{target_match:.4f}\n"
                "첫 불일치 예시:\n"
                f"{mismatch.head(10).to_string(index=False)}"
            )

    # 표준화된 센서/Target은 moldset에서 유지
    drop_cols = [
        SOURCE_INDEX_COL,
        ORIGINAL_INDEX_COL,
        TIMESTAMP_COL,
        PART_COL,
        SIDE_COL,
    ]
    model_part = df.drop(
        columns=[c for c in drop_cols if c in df.columns]
    ).reset_index(drop=True)

    meta = raw_selected[
        [ORIGINAL_INDEX_COL, TIMESTAMP_COL, PART_COL]
    ].reset_index(drop=True)
    meta[SIDE_COL] = derive_side(meta[PART_COL])

    restored = pd.concat([meta, model_part], axis=1)

    unknown_side = int((restored[SIDE_COL] == "UNKNOWN").sum())
    if unknown_side:
        raise ValueError(
            f"PART_NAME에서 RH/LH를 복원하지 못한 행이 {unknown_side}건 있습니다."
        )

    # 인덱스가 실제 RG3를 가리키는지 추가 확인
    non_rg3 = ~restored[PART_COL].astype(str).str.contains(
        "RG3", case=False, na=False
    )
    if non_rg3.any():
        raise ValueError(
            f"복원된 원본 중 RG3가 아닌 행이 {int(non_rg3.sum())}건 있습니다."
        )

    return restored


def add_session_columns(
    df: pd.DataFrame,
    gap_minutes: int = SESSION_GAP_MINUTES,
) -> pd.DataFrame:
    out = df.copy()
    out[TIMESTAMP_COL] = pd.to_datetime(
        out[TIMESTAMP_COL], errors="raise"
    )
    out = out.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    out["Time_Diff"] = out[TIMESTAMP_COL].diff()
    out["New_Session"] = (
        out["Time_Diff"].isna()
        | (out["Time_Diff"] >= pd.Timedelta(minutes=gap_minutes))
    )
    out["Session"] = out["New_Session"].cumsum().astype(int)
    return out


def preprocess_rg3(
    moldset_path: Path = MOLDSET_PATH,
    labeled_data_path: Path = LABELED_DATA_PATH,
    output_path: Path = PREPROCESSED_PATH,
) -> pd.DataFrame:
    moldset_path = Path(moldset_path)
    labeled_data_path = Path(labeled_data_path)
    output_path = Path(output_path)

    if not moldset_path.exists():
        raise FileNotFoundError(
            f"RG3 학습파일이 없습니다: {moldset_path}"
        )
    if not labeled_data_path.exists():
        raise FileNotFoundError(
            f"원본 생산데이터가 없습니다: {labeled_data_path}"
        )

    moldset = pd.read_csv(moldset_path)
    labeled_raw = pd.read_csv(labeled_data_path)

    restored = restore_metadata(moldset, labeled_raw)

    required = [TARGET, TIMESTAMP_COL, PART_COL, SIDE_COL] + RAW_SENSOR_COLS
    missing = [c for c in required if c not in restored.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 없습니다: {missing}")

    # 최종 지도학습은 RH만 사용
    rh = restored.loc[restored[SIDE_COL] == "RH"].copy()

    if rh.empty:
        raise ValueError("RH 데이터가 없습니다.")

    # 데이터 품질 점검
    if rh[SENSOR_COLS + [TARGET]].isna().any().any():
        bad = rh[SENSOR_COLS + [TARGET]].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"결측치가 존재합니다: {bad}")

    numeric = rh[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("공정 센서에 무한대 또는 비수치 값이 있습니다.")

    if rh.duplicated().any():
        raise ValueError(
            f"완전 중복행이 {int(rh.duplicated().sum())}건 있습니다."
        )

    # 상수변수는 모델 입력에서 제거
    rh = rh.drop(
        columns=[c for c in CONSTANT_COLS if c in rh.columns]
    )

    rh = add_session_columns(rh)

    # 보고서/Notebook 기준 구조 검증
    if len(rh) != 591:
        print(
            f"[WARN] RH 데이터 수가 Notebook 기준 591건과 다릅니다: {len(rh)}"
        )

    defect_count = int(rh[TARGET].sum())
    if defect_count != 25:
        print(
            f"[WARN] RH 불량 수가 Notebook 기준 25건과 다릅니다: "
            f"{defect_count}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rh.to_csv(output_path, index=False)

    print("=" * 72)
    print("RG3 PREPROCESS COMPLETE")
    print("=" * 72)
    print(f"Rows          : {len(rh)}")
    print(f"Normal        : {int((rh[TARGET] == 0).sum())}")
    print(f"Defect        : {int((rh[TARGET] == 1).sum())}")
    print(f"Sessions      : {rh['Session'].nunique()}")
    print(f"Sensor cols   : {len(SENSOR_COLS)}")
    print(f"Saved         : {output_path}")

    return rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moldset", type=Path, default=MOLDSET_PATH)
    parser.add_argument("--labeled-data", type=Path, default=LABELED_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PREPROCESSED_PATH)
    args = parser.parse_args()

    preprocess_rg3(
        moldset_path=args.moldset,
        labeled_data_path=args.labeled_data,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
