from src.config_cn7 import (
    INFERENCE_PATH,
    MODEL_INPUT_PATH,
    PREPROCESSED_PATH,
    RAW_MOLDSET_PATH,
    RAW_LABELED_PATH,
)
from src.evaluate_cn7 import evaluate_cn7
from src.feature_engineering_cn7 import build_feature_file
from src.inference_cn7 import inference_cn7
from src.ood_cn7 import train_ood_cn7
from src.preprocess_cn7 import preprocess_file
from src.train_cn7 import train_cn7


def main():
    if not RAW_MOLDSET_PATH.exists():
        raise FileNotFoundError(
            "\nCN7 학습 원본이 없습니다.\n"
            f"다음 위치에 파일을 넣어주세요:\n{RAW_MOLDSET_PATH}\n\n"
            "중요: 현재 노트북에서 학습에 사용한 "
            "`moldset_labeled_cn7_되돌림.csv`처럼 "
            "TimeStamp와 Side(RH_LH/Side/PART_NAME)가 복원된 1,211행 CN7 데이터를 "
            "`moldset_labeled_cn7.csv` 이름으로 두면 됩니다."
        )

    print("\n[1/6] PREPROCESS")
    preprocess_file(
        RAW_MOLDSET_PATH,
        PREPROCESSED_PATH,
        require_target=True,
        raw_labeled_path=RAW_LABELED_PATH,
    )

    print("\n[2/6] FEATURE ENGINEERING")
    build_feature_file(
        PREPROCESSED_PATH,
        MODEL_INPUT_PATH,
    )

    print("\n[3/6] TRAIN + OOF + MODEL SAVE")
    train_cn7()

    print("\n[4/6] EVALUATE")
    evaluate_cn7()

    print("\n[5/6] TRAIN PCA OOD SAFETY LAYER")
    train_ood_cn7()

    print("\n[6/6] INFERENCE SANITY CHECK")
    inference_cn7(
        model_input_path=MODEL_INPUT_PATH,
        output_path=INFERENCE_PATH,
    )

    print("\nCN7 파이프라인 완료")
    print("추가 시간검증 + OOD 재현: python run_cn7_temporal_ood.py")


if __name__ == "__main__":
    main()
