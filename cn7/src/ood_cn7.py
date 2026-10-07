from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .config_cn7 import (
    OOD_CALIBRATION_QUANTILE,
    OOD_FIT_RATIO,
    OOD_MODEL_PATH,
    OOD_PCA_VARIANCE,
    RAW_LABELED_PATH,
    SENSOR_COLS,
    TARGET,
    TIME_COL,
)


def normalize_target(series: pd.Series) -> pd.Series:
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


def prepare_cn7_raw_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """labeled_data.csv에서 CN7 원시 공정행을 복원한다."""
    work = raw_df.copy()
    work.columns = work.columns.astype(str).str.strip()

    required = [TIME_COL, "PART_NAME", TARGET] + SENSOR_COLS
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"CN7 OOD 원본 컬럼 누락: {missing}")

    work = work[
        work["PART_NAME"].astype(str).str.contains("CN7", case=False, na=False)
    ].copy()

    work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="coerce")
    if work[TIME_COL].isna().any():
        raise ValueError("TimeStamp 변환 실패 행이 있습니다.")

    work[TARGET] = normalize_target(work[TARGET])

    for col in SENSOR_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if work[SENSOR_COLS].isna().any().any():
        bad = work[SENSOR_COLS].columns[work[SENSOR_COLS].isna().any()].tolist()
        raise ValueError(f"OOD 센서 숫자 변환/결측 문제: {bad}")

    work["Side"] = np.select(
        [
            work["PART_NAME"].astype(str).str.contains("RH", case=False, na=False),
            work["PART_NAME"].astype(str).str.contains("LH", case=False, na=False),
        ],
        ["RH", "LH"],
        default="UNKNOWN",
    )

    return work.sort_values([TIME_COL, "Side"]).reset_index(drop=True)


def build_cycle_frame(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """
    원시 CN7을 생산 Cycle(TimeStamp) 단위로 압축한다.

    같은 TimeStamp의 센서값은 공정 Cycle 공통값이므로 median을 사용하고,
    LH/RH 중 하나라도 불량이면 해당 Cycle을 불량 Cycle로 정의한다.
    """
    cycle = (
        raw_rows.groupby(TIME_COL, as_index=False)
        .agg({**{c: "median" for c in SENSOR_COLS}, TARGET: "max"})
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    return cycle


def reconstruction_error(scaler: RobustScaler, pca: PCA, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    arr = scaler.transform(x)
    z = pca.transform(arr)
    x_hat = pca.inverse_transform(z)
    return np.mean((arr - x_hat) ** 2, axis=1)


def fit_ood_artifact(
    normal_cycle_df: pd.DataFrame,
    fit_ratio: float = OOD_FIT_RATIO,
    calibration_quantile: float = OOD_CALIBRATION_QUANTILE,
    pca_variance: float = OOD_PCA_VARIANCE,
) -> dict:
    """
    과거 정상 Cycle만으로 PCA Reconstruction Error OOD 모델을 만든다.

    시간순 앞부분은 PCA fit, 뒷부분은 threshold calibration에 사용한다.
    미래 불량/미래 정상 라벨은 threshold 결정에 사용하지 않는다.
    """
    normal = normal_cycle_df.sort_values(TIME_COL).reset_index(drop=True).copy()
    if TARGET in normal.columns:
        normal = normal[normal[TARGET].astype(int) == 0].reset_index(drop=True)

    if len(normal) < 20:
        raise ValueError(f"OOD 정상 Cycle이 너무 적습니다: {len(normal)}")

    split_idx = int(len(normal) * fit_ratio)
    split_idx = min(max(split_idx, 1), len(normal) - 1)

    fit_df = normal.iloc[:split_idx].copy()
    cal_df = normal.iloc[split_idx:].copy()

    scaler = RobustScaler()
    x_fit = scaler.fit_transform(fit_df[SENSOR_COLS])

    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=42)
    pca.fit(x_fit)

    cal_scores = reconstruction_error(scaler, pca, cal_df[SENSOR_COLS])
    threshold = float(np.quantile(cal_scores, calibration_quantile))
    cal_flags = cal_scores > threshold

    return {
        "method": "PCA_Reconstruction_Error",
        "sensor_cols": list(SENSOR_COLS),
        "scaler": scaler,
        "pca": pca,
        "threshold": threshold,
        "fit_ratio": float(fit_ratio),
        "calibration_quantile": float(calibration_quantile),
        "pca_variance": float(pca_variance),
        "fit_cycle_count": int(len(fit_df)),
        "calibration_cycle_count": int(len(cal_df)),
        "calibration_ood_rate": float(cal_flags.mean()),
        "train_start": str(normal[TIME_COL].min()),
        "train_end": str(normal[TIME_COL].max()),
    }


def score_ood_cycles(cycle_df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    missing = [c for c in artifact["sensor_cols"] if c not in cycle_df.columns]
    if missing:
        raise ValueError(f"OOD 추론 센서 컬럼 누락: {missing}")

    result = cycle_df.copy()
    scores = reconstruction_error(
        artifact["scaler"],
        artifact["pca"],
        result[artifact["sensor_cols"]],
    )
    result["OOD_Score"] = scores
    result["OOD_Threshold"] = float(artifact["threshold"])

    # OOD baseline은 artifact의 학습 종료 이후 신규 Cycle에만 적용한다.
    # 과거 데이터를 현재 baseline과 비교해 대량 OOD로 오해하는 것을 방지한다.
    train_end = pd.Timestamp(artifact["train_end"])
    result["OOD_Applicable"] = (pd.to_datetime(result[TIME_COL]) > train_end).astype(int)
    result["OOD_Flag"] = (
        (scores > artifact["threshold"])
        & (result["OOD_Applicable"] == 1)
    ).astype(int)
    return result


def train_ood_cn7(
    raw_labeled_path=RAW_LABELED_PATH,
    model_path=OOD_MODEL_PATH,
) -> dict:
    """
    운영용 OOD baseline을 생성한다.

    supervised CN7 학습과 같은 과거 구간(2020-10-16/19/20)만 사용하며,
    정상 Cycle만으로 PCA를 학습한다. 이는 E4 이후 정보를 보지 않는 baseline이다.
    """
    raw = pd.read_csv(raw_labeled_path)
    rows = prepare_cn7_raw_rows(raw)

    train_dates = {"2020-10-16", "2020-10-19", "2020-10-20"}
    train_rows = rows[
        rows[TIME_COL].dt.strftime("%Y-%m-%d").isin(train_dates)
    ].copy()
    cycles = build_cycle_frame(train_rows)
    normal_cycles = cycles[cycles[TARGET] == 0].copy()

    artifact = fit_ood_artifact(normal_cycles)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    print("\n[CN7 OOD SAFETY LAYER]")
    print(f"Normal cycles        : {len(normal_cycles):,}")
    print(f"OOD fit cycles       : {artifact['fit_cycle_count']:,}")
    print(f"OOD calibration      : {artifact['calibration_cycle_count']:,}")
    print(f"PCA components       : {artifact['pca'].n_components_}")
    print(f"OOD threshold        : {artifact['threshold']:.6f}")
    print(f"Calibration OOD rate : {artifact['calibration_ood_rate'] * 100:.3f}%")
    print(f"OOD model saved      : {model_path}")

    return artifact


def load_ood_artifact(model_path=OOD_MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"OOD 모델 파일이 없습니다: {model_path}\n"
            "먼저 python run_cn7.py 를 실행하세요."
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    train_ood_cn7()
