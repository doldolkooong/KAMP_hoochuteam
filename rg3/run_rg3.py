import argparse
from pathlib import Path


from src.config_rg3 import (
    LABELED_DATA_PATH,
    MODEL_INPUT_PATH,
    MODEL_PATH,
    MOLDSET_PATH,
    PREPROCESSED_PATH,
    TEST_PATH,
    UNLABELED_DATA_PATH,
)

from src.evaluate_rg3 import (
    run_evaluation,
)

from src.feature_engineering_rg3 import (
    build_model_input,
)

from src.inference_rg3 import (
    inference_rg3_test,
)

from src.preprocess_rg3 import (
    preprocess_rg3,
)

from src.train_rg3 import (
    train_final_model,
)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RG3 전체 재현 Pipeline"
        )
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    parser.add_argument(
        "--moldset",
        type=Path,
        default=MOLDSET_PATH,
    )


    parser.add_argument(
        "--labeled-data",
        type=Path,
        default=LABELED_DATA_PATH,
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    parser.add_argument(
        "--test",
        type=Path,
        default=TEST_PATH,
    )


    parser.add_argument(
        "--unlabeled-data",
        type=Path,
        default=UNLABELED_DATA_PATH,
    )


    # --------------------------------------------------------
    # Test Option
    # --------------------------------------------------------

    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help=(
            "Test 우선검사 Top-N. "
            "0이면 Risk Ranking만 생성"
        ),
    )


    parser.add_argument(
        "--skip-test",
        action="store_true",
        help=(
            "Test 추론을 수행하지 않음"
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # 1. PREPROCESS
    # ========================================================

    print()

    print(
        "[1/5] Preprocess"
    )


    preprocess_rg3(
        moldset_path=args.moldset,
        labeled_data_path=args.labeled_data,
        output_path=PREPROCESSED_PATH,
    )


    # ========================================================
    # 2. FEATURE ENGINEERING
    # ========================================================

    print()

    print(
        "[2/5] Feature Engineering"
    )


    build_model_input(
        input_path=PREPROCESSED_PATH,
        output_path=MODEL_INPUT_PATH,
    )


    # ========================================================
    # 3. EVALUATION
    # ========================================================

    print()

    print(
        "[3/5] Temporal / Holdout Evaluation"
    )


    run_evaluation(
        input_path=MODEL_INPUT_PATH,
    )


    # ========================================================
    # 4. FINAL MODEL
    # ========================================================

    print()

    print(
        "[4/5] Final Model Training"
    )


    train_final_model(
        input_path=MODEL_INPUT_PATH,
        model_path=MODEL_PATH,
    )


    # ========================================================
    # 5. TEST INFERENCE
    # ========================================================

    if args.skip_test:

        print()

        print(
            "[5/5] Test Inference SKIPPED"
        )

    else:

        print()

        print(
            "[5/5] Test Inference"
        )


        if not args.test.exists():

            raise FileNotFoundError(
                "Test Moldset이 없습니다:\n"
                f"{args.test}"
            )


        if not args.unlabeled_data.exists():

            raise FileNotFoundError(
                "unlabeled_data.csv가 없습니다:\n"
                f"{args.unlabeled_data}"
            )


        inference_rg3_test(
            test_path=args.test,
            unlabeled_data_path=args.unlabeled_data,
            model_path=MODEL_PATH,
            top_n=args.top_n,
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "RG3 PIPELINE COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()