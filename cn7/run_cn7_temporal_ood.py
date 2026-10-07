from src.temporal_ood_cn7 import evaluate_temporal_ood


def main():
    print("=" * 80)
    print("CN7 TEMPORAL VALIDATION + PCA OOD SAFETY LAYER")
    print("=" * 80)
    evaluate_temporal_ood()


if __name__ == "__main__":
    main()
