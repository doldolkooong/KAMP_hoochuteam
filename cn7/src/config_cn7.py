from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "data" / "raw"
SHARED_RAW_DIR = ROOT_DIR.parent / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

RAW_MOLDSET_PATH = RAW_DIR / "moldset_labeled_cn7.csv"
RAW_LABELED_PATH = SHARED_RAW_DIR / "labeled_data.csv"

PREPROCESSED_PATH = PROCESSED_DIR / "train_cn7_preprocessed.csv"
MODEL_INPUT_PATH = PROCESSED_DIR / "cn7_model_input.csv"

OOF_PATH = OUTPUT_DIR / "cn7_oof_prediction.csv"
THRESHOLD_PATH = OUTPUT_DIR / "cn7_threshold_result.csv"
INFERENCE_PATH = OUTPUT_DIR / "cn7_inference_result.csv"
METRICS_PATH = OUTPUT_DIR / "cn7_metrics.csv"

OOD_MODEL_PATH = MODEL_DIR / "cn7_pca_ood.pkl"
OOD_TEMPORAL_SIDE_RESULTS_PATH = OUTPUT_DIR / "cn7_temporal_ood_side_results.csv"
OOD_TEMPORAL_CYCLE_RESULTS_PATH = OUTPUT_DIR / "cn7_temporal_ood_cycle_results.csv"
OOD_TEMPORAL_PREDICTIONS_PATH = OUTPUT_DIR / "cn7_temporal_ood_predictions.csv"

TARGET = "PassOrFail"
TIME_COL = "TimeStamp"
SIDE_FEATURE = "Side_RH"

SENSOR_COLS = [
    "Injection_Time",
    "Filling_Time",
    "Plasticizing_Time",
    "Cycle_Time",
    "Clamp_Close_Time",
    "Cushion_Position",
    "Plasticizing_Position",
    "Max_Injection_Speed",
    "Max_Screw_RPM",
    "Average_Screw_RPM",
    "Max_Injection_Pressure",
    "Max_Switch_Over_Pressure",
    "Max_Back_Pressure",
    "Average_Back_Pressure",
    "Barrel_Temperature_1",
    "Barrel_Temperature_2",
    "Barrel_Temperature_3",
    "Barrel_Temperature_4",
    "Barrel_Temperature_5",
    "Barrel_Temperature_6",
    "Hopper_Temperature",
    "Mold_Temperature_3",
    "Mold_Temperature_4",
]

ROLL_WINDOW = 5
SESSION_GAP_SECONDS = 300

ROLL_MEAN_COLS = [f"{c}_roll5_mean" for c in SENSOR_COLS]
FEATURE_COLS = SENSOR_COLS + [SIDE_FEATURE] + ROLL_MEAN_COLS

CV_SEEDS = [42, 100, 2026, 777, 1234]
N_SPLITS = 4
CV_MODEL_RANDOM_STATE = 42

BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.10,
    "num_leaves": 15,
    "max_depth": 7,
    "min_child_samples": 10,
    "subsample": 1.0,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}

CLASS_WEIGHT = "balanced"

F1_THRESHOLD = 0.080
OPERATION_THRESHOLD = 0.060
HIGH_PRECISION_THRESHOLD = 0.470
HIGH_RECALL_THRESHOLD = 0.020

# PCA OOD Safety Layer
# - 과거 정상 Cycle만 사용
# - 시간순 80% fit / 20% calibration
# - calibration 정상 score의 99 percentile을 threshold로 사용
OOD_FIT_RATIO = 0.80
OOD_CALIBRATION_QUANTILE = 0.99
OOD_PCA_VARIANCE = 0.95

EXPECTED_INTERNAL_PR_AUC_MEAN = 0.8157169084660907
EXPECTED_INTERNAL_PR_AUC_STD = 0.013893581153823713

# TimeStamp/Side 복원에 사용하는 24개 원본 공정변수
# (최종 모델에서는 Clamp_Open_Position 제거)
RESTORE_FEATURE_COLS = [
    "Injection_Time",
    "Filling_Time",
    "Plasticizing_Time",
    "Cycle_Time",
    "Clamp_Close_Time",
    "Cushion_Position",
    "Plasticizing_Position",
    "Clamp_Open_Position",
    "Max_Injection_Speed",
    "Max_Screw_RPM",
    "Average_Screw_RPM",
    "Max_Injection_Pressure",
    "Max_Switch_Over_Pressure",
    "Max_Back_Pressure",
    "Average_Back_Pressure",
    "Barrel_Temperature_1",
    "Barrel_Temperature_2",
    "Barrel_Temperature_3",
    "Barrel_Temperature_4",
    "Barrel_Temperature_5",
    "Barrel_Temperature_6",
    "Hopper_Temperature",
    "Mold_Temperature_3",
    "Mold_Temperature_4",
]
