from pathlib import Path


class Config:
    RANDOM_SEED = 24
    TEST_SIZE = .2
    ASSETS_PATH = Path("./assets")
    ORIGINAL_DATASET_PATH = ASSETS_PATH / "original_dataset"
    ORIGINAL_DATASET_FILE_PATH = ORIGINAL_DATASET_PATH / "data.csv"
    DATASET_PATH = ASSETS_PATH / "data"
    FEATURES_PATH = ASSETS_PATH / "features"
    MODELS_PATH = ASSETS_PATH / "models"
    MODELS_V2_PATH = ASSETS_PATH / "models_v2"
    METRICS_FILE_PATH = ASSETS_PATH / "metrics.json"
    METRICS_V2_FILE_PATH = ASSETS_PATH / "metrics_v2.json"
