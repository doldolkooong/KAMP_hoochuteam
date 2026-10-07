import pandas as pd

from .config_cn7 import (
    FEATURE_COLS,
    ROLL_MEAN_COLS,
    ROLL_WINDOW,
    SENSOR_COLS,
    SESSION_GAP_SECONDS,
    SIDE_FEATURE,
    TARGET,
    TIME_COL,
)


def _check_cycle_sensor_consistency(df: pd.DataFrame) -> None:
    """동일 TimeStamp의 LH/RH가 서로 다른 센서 조합을 갖는지 검사한다."""
    unique_sensor_rows = (
        df[[TIME_COL] + SENSOR_COLS]
        .drop_duplicates()
        .groupby(TIME_COL)
        .size()
    )
    conflicts = unique_sensor_rows[unique_sensor_rows > 1]

    if len(conflicts) > 0:
        raise ValueError(
            f"동일 TimeStamp 내부 센서 조합이 다른 Cycle이 {len(conflicts)}개 있습니다. "
            "기존 47-feature recipe를 그대로 적용할 수 없습니다."
        )


def create_cn7_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    최종 V5 Feature를 생성한다.

    23 Original Sensor
    + Side_RH 1
    + 직전 5 Cycle Rolling Mean 23
    = 47 Features

    rolling은 반드시 shift(1) 후 계산하여 현재 Cycle 정보를 사용하지 않는다.
    5분 초과 생산 공백은 새 session으로 간주한다.
    """
    work = df.copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors="raise")
    work = work.sort_values([TIME_COL, SIDE_FEATURE]).reset_index(drop=True)

    _check_cycle_sensor_consistency(work)

    cycle = (
        work[[TIME_COL] + SENSOR_COLS]
        .drop_duplicates(subset=[TIME_COL], keep="first")
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    cycle["_time_gap_sec"] = cycle[TIME_COL].diff().dt.total_seconds()
    cycle["_session_id"] = cycle["_time_gap_sec"].gt(SESSION_GAP_SECONDS).cumsum()

    for sensor, roll_col in zip(SENSOR_COLS, ROLL_MEAN_COLS):
        cycle[roll_col] = (
            cycle.groupby("_session_id")[sensor]
            .transform(
                lambda s: s.shift(1).rolling(
                    window=ROLL_WINDOW,
                    min_periods=1,
                ).mean()
            )
            .fillna(0)
        )

    result = work.merge(
        cycle[[TIME_COL] + ROLL_MEAN_COLS],
        on=TIME_COL,
        how="left",
        validate="many_to_one",
    )

    if result[ROLL_MEAN_COLS].isna().any().any():
        raise ValueError("Rolling Mean 병합 후 결측치가 발생했습니다.")

    if len(FEATURE_COLS) != 47:
        raise RuntimeError(f"최종 Feature 수가 47이 아닙니다: {len(FEATURE_COLS)}")

    columns = [TIME_COL]
    if TARGET in result.columns:
        columns.append(TARGET)
    columns += FEATURE_COLS

    return result[columns].copy()


def build_feature_file(input_path, output_path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    result = create_cn7_features(df)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    from .config_cn7 import MODEL_INPUT_PATH, PREPROCESSED_PATH

    MODEL_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = build_feature_file(PREPROCESSED_PATH, MODEL_INPUT_PATH)
    print(f"저장 완료: {MODEL_INPUT_PATH}")
    print("Shape:", result.shape)
    print("Feature 수:", len(FEATURE_COLS))
