from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DIRECTORY
# ============================================================

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SHARED_RAW_DIR = PROJECT_ROOT.parent / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# RAW DATA
# ============================================================

# Train
MOLDSET_PATH = (
    RAW_DIR
    / "moldset_labeled_rg3.csv"
)

LABELED_DATA_PATH = (
    SHARED_RAW_DIR
    / "labeled_data.csv"
)


# Test
TEST_PATH = (
    RAW_DIR
    / "moldset_unlabeled_rg3.csv"
)

UNLABELED_DATA_PATH = (
    SHARED_RAW_DIR
    / "unlabeled_data.csv"
)


# ============================================================
# PROCESSED DATA
# ============================================================

PREPROCESSED_PATH = (
    PROCESSED_DIR
    / "train_rg3_preprocessed.csv"
)

MODEL_INPUT_PATH = (
    PROCESSED_DIR
    / "rg3_model_input.csv"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "rg3_xgb_final.pkl"
)


# ============================================================
# OUTPUT
# ============================================================

TEMPORAL_RESULT_PATH = (
    OUTPUT_DIR
    / "rg3_temporal_validation.csv"
)

HOLDOUT_PREDICTION_PATH = (
    OUTPUT_DIR
    / "rg3_holdout_prediction.csv"
)

RISK_PREDICTION_PATH = (
    OUTPUT_DIR
    / "rg3_risk_prediction.csv"
)


# ============================================================
# COLUMN
# ============================================================

TARGET = "PassOrFail"

TIMESTAMP_COL = "TimeStamp"

PART_COL = "PART_NAME"

SIDE_COL = "RH_LH"

ORIGINAL_INDEX_COL = "original_index"

SOURCE_INDEX_COL = "Unnamed: 0"


# ============================================================
# ORIGINAL SENSOR
# ============================================================

RAW_SENSOR_COLS = [
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


# ============================================================
# CONSTANT FEATURE
# ============================================================

CONSTANT_COLS = [
    "Clamp_Open_Position",
]


# 실제 모델에서 사용하는 23개 센서
SENSOR_COLS = [
    col
    for col in RAW_SENSOR_COLS
    if col not in CONSTANT_COLS
]


# ============================================================
# SESSION
# ============================================================

SESSION_GAP_MINUTES = 30


# ============================================================
# EWMA
# ============================================================

EWMA_SPAN = 5


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# FINAL MODEL FEATURE
# ============================================================

MODEL_FEATURES = [
    f"{col}_ewma{EWMA_SPAN}_abs_dev"
    for col in SENSOR_COLS
]


# ============================================================
# XGBOOST PARAMETER
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}
