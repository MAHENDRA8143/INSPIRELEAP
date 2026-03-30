from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "diabetes"

MODEL_FEATURES = ["age", "bmi", "glucose", "cholesterol", "systolic_bp", "physical_activity_level"]
