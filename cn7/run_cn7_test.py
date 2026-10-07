from src.test_cn7 import run_test_pipeline


def main():
    print("=" * 80)
    print("CN7 TEST INFERENCE + PCA OOD SAFETY LAYER")
    print("=" * 80)
    run_test_pipeline()
    print("\nCN7 테스트 추론 완료")


if __name__ == "__main__":
    main()
