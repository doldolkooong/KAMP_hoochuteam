from pathlib import Path

import joblib
import numpy as np
import pandas as pd


from src.config_rg3 import (
    MODEL_FEATURES,
    MODEL_PATH,
    RISK_PREDICTION_PATH,
    SENSOR_COLS,
    SIDE_COL,
    TEST_PATH,
    TIMESTAMP_COL,
    UNLABELED_DATA_PATH,
)

from src.preprocess_rg3 import (
    add_session_columns,
)

from src.feature_engineering_rg3 import (
    add_absolute_ewma_features,
)


# ============================================================
# RH / LH 복원
# ============================================================

def derive_side(
    part_name: pd.Series,
) -> pd.Series:

    part = (
        part_name
        .astype(str)
    )

    side = np.select(
        [
            part.str.contains(
                "RH",
                case=False,
                na=False,
            ),
            part.str.contains(
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

    return pd.Series(
        side,
        index=part_name.index,
        name=SIDE_COL,
    )


# ============================================================
# TEST METADATA 복원
# ============================================================

def restore_test_metadata(
    test: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:

    """
    moldset_unlabeled_rg3.csv의 Unnamed: 0을 이용하여
    unlabeled_data.csv에서 생산 메타정보를 복원한다.

    복원 대상
    ----------
    - original_index
    - TimeStamp
    - PART_NAME
    - RH_LH

    주의
    ----
    Test Label은 사용하지 않는다.
    """

    # --------------------------------------------------------
    # 이미 복원된 데이터라면 그대로 사용
    # --------------------------------------------------------

    if (
        TIMESTAMP_COL in test.columns
        and "PART_NAME" in test.columns
        and SIDE_COL in test.columns
    ):

        restored = test.copy()

        if "original_index" not in restored.columns:

            if "Unnamed: 0" in restored.columns:

                restored[
                    "original_index"
                ] = pd.to_numeric(
                    restored["Unnamed: 0"],
                    errors="raise",
                ).astype(int)

            else:

                restored[
                    "original_index"
                ] = np.arange(
                    len(restored)
                )

        return restored


    # --------------------------------------------------------
    # 원본 index 존재 확인
    # --------------------------------------------------------

    if "Unnamed: 0" not in test.columns:

        raise KeyError(
            "moldset_unlabeled_rg3.csv에 "
            "'Unnamed: 0' 컬럼이 없습니다."
        )


    source_index = pd.to_numeric(
        test["Unnamed: 0"],
        errors="raise",
    ).astype(int)


    print(
        "Source index range:",
        source_index.min(),
        "~",
        source_index.max(),
    )

    print(
        "unlabeled_data rows:",
        len(raw),
    )


    # --------------------------------------------------------
    # 범위 검사
    # --------------------------------------------------------

    if source_index.min() < 0:

        raise IndexError(
            "Unnamed: 0에 음수 인덱스가 존재합니다."
        )


    if source_index.max() >= len(raw):

        raise IndexError(
            "\n"
            "moldset_unlabeled_rg3.csv의 원본 인덱스가\n"
            "unlabeled_data.csv 범위를 벗어났습니다.\n\n"
            f"Source index max : {source_index.max()}\n"
            f"Raw rows         : {len(raw)}\n"
        )


    # --------------------------------------------------------
    # 원본 행 복원
    # --------------------------------------------------------

    raw_selected = (
        raw
        .iloc[
            source_index.to_numpy()
        ]
        .copy()
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "original_index",
            }
        )
    )


    # --------------------------------------------------------
    # 필요한 Metadata 확인
    # --------------------------------------------------------

    required_meta = [
        TIMESTAMP_COL,
        "PART_NAME",
    ]

    missing_meta = [
        col
        for col in required_meta
        if col not in raw_selected.columns
    ]

    if missing_meta:

        raise KeyError(
            "unlabeled_data.csv에 필요한 "
            f"Metadata가 없습니다: {missing_meta}"
        )


    # --------------------------------------------------------
    # Metadata만 추출
    # --------------------------------------------------------

    meta = raw_selected[
        [
            "original_index",
            TIMESTAMP_COL,
            "PART_NAME",
        ]
    ].copy()


    # --------------------------------------------------------
    # RH / LH 생성
    # --------------------------------------------------------

    meta[SIDE_COL] = derive_side(
        meta["PART_NAME"]
    )


    unknown_count = int(
        (
            meta[SIDE_COL]
            == "UNKNOWN"
        ).sum()
    )


    if unknown_count > 0:

        raise ValueError(
            "PART_NAME에서 RH/LH를 복원하지 못한 "
            f"데이터가 {unknown_count}건 있습니다."
        )


    # --------------------------------------------------------
    # RG3 데이터인지 확인
    # --------------------------------------------------------

    is_rg3 = (
        meta["PART_NAME"]
        .astype(str)
        .str.contains(
            "RG3",
            case=False,
            na=False,
        )
    )


    if not is_rg3.all():

        bad_count = int(
            (~is_rg3).sum()
        )

        bad_example = (
            meta.loc[
                ~is_rg3,
                "PART_NAME",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "복원된 데이터 중 RG3가 아닌 데이터가 "
            f"{bad_count}건 있습니다.\n"
            f"예시: {bad_example}"
        )


    # --------------------------------------------------------
    # Moldset 센서 데이터
    # --------------------------------------------------------

    drop_cols = [
        "Unnamed: 0",
        "original_index",
        TIMESTAMP_COL,
        "PART_NAME",
        SIDE_COL,
        "PassOrFail",
    ]


    model_data = (
        test
        .drop(
            columns=[
                col
                for col in drop_cols
                if col in test.columns
            ]
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Metadata + Scaling된 Sensor 결합
    # --------------------------------------------------------

    restored = pd.concat(
        [
            meta.reset_index(drop=True),
            model_data,
        ],
        axis=1,
    )


    return restored


# ============================================================
# TEST DATA 준비
# ============================================================

def prepare_test_data(
    test_path: Path = TEST_PATH,
    unlabeled_data_path: Path = UNLABELED_DATA_PATH,
) -> pd.DataFrame:

    test_path = Path(
        test_path
    )

    unlabeled_data_path = Path(
        unlabeled_data_path
    )


    # --------------------------------------------------------
    # 파일 확인
    # --------------------------------------------------------

    if not test_path.exists():

        raise FileNotFoundError(
            f"Test 파일이 없습니다:\n{test_path}"
        )


    if not unlabeled_data_path.exists():

        raise FileNotFoundError(
            "unlabeled_data.csv가 없습니다.\n"
            f"{unlabeled_data_path}"
        )


    # --------------------------------------------------------
    # Test load
    # --------------------------------------------------------

    test = pd.read_csv(
        test_path
    )


    raw = pd.read_csv(
        unlabeled_data_path
    )


    print(
        "=" * 80
    )

    print(
        "RG3 TEST DATA"
    )

    print(
        "=" * 80
    )

    print(
        "Moldset shape:",
        test.shape,
    )

    print(
        "Raw unlabeled shape:",
        raw.shape,
    )


    # --------------------------------------------------------
    # Metadata 복원
    # --------------------------------------------------------

    test = restore_test_metadata(
        test=test,
        raw=raw,
    )


    # --------------------------------------------------------
    # Target 사용 금지
    # --------------------------------------------------------

    if "PassOrFail" in test.columns:

        test = test.drop(
            columns=[
                "PassOrFail"
            ]
        )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    test[
        TIMESTAMP_COL
    ] = pd.to_datetime(
        test[
            TIMESTAMP_COL
        ],
        errors="raise",
    )


    # --------------------------------------------------------
    # 센서 확인
    # --------------------------------------------------------

    missing_sensor = [
        col
        for col in SENSOR_COLS
        if col not in test.columns
    ]


    if missing_sensor:

        raise KeyError(
            "Test 데이터에 필요한 센서가 없습니다:\n"
            f"{missing_sensor}"
        )


    # --------------------------------------------------------
    # 숫자 변환 확인
    # --------------------------------------------------------

    for col in SENSOR_COLS:

        test[col] = pd.to_numeric(
            test[col],
            errors="raise",
        )


    # --------------------------------------------------------
    # 결과 확인
    # --------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "METADATA RESTORE COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        test[
            [
                "original_index",
                TIMESTAMP_COL,
                "PART_NAME",
                SIDE_COL,
            ]
        ]
        .head()
        .to_string(
            index=False
        )
    )


    print()

    print(
        "Side distribution"
    )

    print(
        test[
            SIDE_COL
        ].value_counts()
    )


    return test


# ============================================================
# TEST INFERENCE
# ============================================================

def inference_rg3_test(
    test_path: Path = TEST_PATH,
    unlabeled_data_path: Path = UNLABELED_DATA_PATH,
    model_path: Path = MODEL_PATH,
    output_path: Path = RISK_PREDICTION_PATH,
    top_n: int = 0,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # 1. Test 준비
    # --------------------------------------------------------

    test = prepare_test_data(
        test_path=test_path,
        unlabeled_data_path=unlabeled_data_path,
    )


    # 원래 Moldset 순서 저장
    test[
        "_input_order"
    ] = np.arange(
        len(test)
    )


    # --------------------------------------------------------
    # 2. RH / LH 분리
    # --------------------------------------------------------

    rh = (
        test[
            test[SIDE_COL] == "RH"
        ]
        .copy()
    )


    lh = (
        test[
            test[SIDE_COL] == "LH"
        ]
        .copy()
    )


    print()

    print(
        "=" * 80
    )

    print(
        "SIDE DISTRIBUTION"
    )

    print(
        "=" * 80
    )

    print(
        "RH:",
        len(rh),
    )

    print(
        "LH:",
        len(lh),
    )


    # --------------------------------------------------------
    # 3. 최종 모델 로드
    # --------------------------------------------------------

    model_path = Path(
        model_path
    )


    if not model_path.exists():

        raise FileNotFoundError(
            "최종 RG3 모델이 없습니다:\n"
            f"{model_path}"
        )


    bundle = joblib.load(
        model_path
    )


    # 저장 형태가 bundle인 경우
    if isinstance(
        bundle,
        dict,
    ):

        model = bundle[
            "model"
        ]

        feature_names = bundle.get(
            "feature_names",
            MODEL_FEATURES,
        )

    else:

        # 혹시 모델 객체만 저장한 경우
        model = bundle

        feature_names = (
            MODEL_FEATURES
        )


    result_list = []


    # ========================================================
    # 4. RH Risk Ranking
    # ========================================================

    if not rh.empty:

        # ----------------------------------------------------
        # 생산시간 정렬 + Session 생성
        # ----------------------------------------------------

        rh = add_session_columns(
            rh
        )


        print()

        print(
            "=" * 80
        )

        print(
            "RH SESSION"
        )

        print(
            "=" * 80
        )

        print(
            "Session count:",
            rh["Session"].nunique(),
        )


        session_summary = (
            rh.groupby(
                "Session"
            )
            .agg(
                Start_Time=(
                    TIMESTAMP_COL,
                    "min",
                ),
                End_Time=(
                    TIMESTAMP_COL,
                    "max",
                ),
                Rows=(
                    TIMESTAMP_COL,
                    "size",
                ),
            )
            .reset_index()
        )


        print(
            session_summary.to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # Absolute EWMA5 생성
        # ----------------------------------------------------

        rh = add_absolute_ewma_features(
            rh
        )


        # ----------------------------------------------------
        # 최종 Feature 확인
        # ----------------------------------------------------

        missing_features = [
            col
            for col in feature_names
            if col not in rh.columns
        ]


        if missing_features:

            raise KeyError(
                "최종 모델 Feature를 생성하지 못했습니다:\n"
                f"{missing_features}"
            )


        X_test = rh[
            feature_names
        ].copy()


        print()

        print(
            "=" * 80
        )

        print(
            "MODEL INPUT"
        )

        print(
            "=" * 80
        )

        print(
            "RH rows:",
            len(X_test),
        )

        print(
            "Features:",
            len(feature_names),
        )

        print(
            "NaN count:",
            int(
                X_test
                .isna()
                .sum()
                .sum()
            ),
        )


        # ----------------------------------------------------
        # Risk Score
        # ----------------------------------------------------

        rh[
            "Risk_Score"
        ] = model.predict_proba(
            X_test
        )[:, 1]


        # ----------------------------------------------------
        # Risk Rank
        # ----------------------------------------------------

        rh[
            "Risk_Rank"
        ] = (
            rh[
                "Risk_Score"
            ]
            .rank(
                method="first",
                ascending=False,
            )
            .astype(int)
        )


        # ----------------------------------------------------
        # AI Status
        # ----------------------------------------------------

        rh[
            "AI_Status"
        ] = (
            "RH_RISK_RANKING"
        )


        # ----------------------------------------------------
        # Inspection Priority
        # ----------------------------------------------------

        if top_n > 0:

            rh[
                "Inspection_Priority"
            ] = np.where(
                rh[
                    "Risk_Rank"
                ] <= top_n,
                "PRIORITY_INSPECTION",
                "EXISTING_INSPECTION",
            )

        else:

            # 검증되지 않은 고정 Threshold를 임의 적용하지 않음
            rh[
                "Inspection_Priority"
            ] = (
                "RANK_BY_RISK_SCORE"
            )


        result_list.append(
            rh
        )


    # ========================================================
    # 5. LH
    # ========================================================

    if not lh.empty:

        lh[
            "Session"
        ] = pd.NA

        lh[
            "Risk_Score"
        ] = np.nan

        lh[
            "Risk_Rank"
        ] = pd.NA

        lh[
            "AI_Status"
        ] = (
            "LH_EXCLUDED_NO_DEFECT_TRAINING"
        )

        lh[
            "Inspection_Priority"
        ] = (
            "EXISTING_INSPECTION"
        )


        result_list.append(
            lh
        )


    # --------------------------------------------------------
    # 데이터 존재 여부
    # --------------------------------------------------------

    if not result_list:

        raise ValueError(
            "RH 또는 LH 데이터가 없습니다."
        )


    # ========================================================
    # 6. RH + LH 다시 결합
    # ========================================================

    result = pd.concat(
        result_list,
        ignore_index=True,
    )


    # 원래 Test 순서 복원
    result = (
        result
        .sort_values(
            "_input_order"
        )
        .drop(
            columns=[
                "_input_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 7. 출력 컬럼 순서
    # ========================================================

    first_cols = [
        "original_index",
        TIMESTAMP_COL,
        "PART_NAME",
        SIDE_COL,
        "Session",
        "Risk_Score",
        "Risk_Rank",
        "Inspection_Priority",
        "AI_Status",
    ]


    first_cols = [
        col
        for col in first_cols
        if col in result.columns
    ]


    other_cols = [
        col
        for col in result.columns
        if col not in first_cols
    ]


    result = result[
        first_cols
        + other_cols
    ]


    # ========================================================
    # 8. 저장
    # ========================================================

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    result.to_csv(
        output_path,
        index=False,
    )


    # ========================================================
    # 9. 결과 요약
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "RG3 TEST INFERENCE COMPLETE"
    )

    print(
        "=" * 80
    )


    print(
        "Total:",
        len(result),
    )


    print(
        "RH:",
        int(
            (
                result[SIDE_COL]
                == "RH"
            ).sum()
        ),
    )


    print(
        "LH:",
        int(
            (
                result[SIDE_COL]
                == "LH"
            ).sum()
        ),
    )


    # --------------------------------------------------------
    # RH Risk 통계
    # --------------------------------------------------------

    if not rh.empty:

        print()

        print(
            "RH Risk Score"
        )

        print(
            rh[
                "Risk_Score"
            ]
            .describe()
            .to_string()
        )


        print()

        print(
            "=" * 80
        )

        print(
            "TOP 20 RISK"
        )

        print(
            "=" * 80
        )


        show_cols = [
            col
            for col in [
                "original_index",
                TIMESTAMP_COL,
                "PART_NAME",
                SIDE_COL,
                "Session",
                "Risk_Score",
                "Risk_Rank",
            ]
            if col in rh.columns
        ]


        print(
            rh
            .sort_values(
                "Risk_Score",
                ascending=False,
            )
            [show_cols]
            .head(20)
            .to_string(
                index=False
            )
        )


    print()

    print(
        "Saved:",
        output_path,
    )


    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    inference_rg3_test()