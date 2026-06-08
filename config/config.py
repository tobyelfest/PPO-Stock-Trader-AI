import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "datasets"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FEATURED_DATA_DIR = DATA_DIR / "featured"
    MODELS_DIR = BASE_DIR / "models_store"
    TRACKING_DIR = BASE_DIR / "tracking_data"
    MLRUNS_DIR = TRACKING_DIR / "mlruns"
    ARTIFACTS_DIR = TRACKING_DIR / "artifacts"

    # Data parameters
    STOCKS: List[str] = ["BBCA.JK"]
    START_DATE: str = "2018-01-01"
    END_DATE: str = "2023-12-31"
    VALIDATION_SPLIT_DATE: str = "2022-01-01"
    TEST_SPLIT_DATE: str = "2023-01-01"

    # Technical indicators
    INDICATORS: List[str] = [
        "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
        "macd", "macd_signal", "bb_upper", "bb_lower", "atr_14"
    ]

    # Trading environment
    INITIAL_CAPITAL: float = 100_000.0
    TRANSACTION_COST_PCT: float = 0.001  # 0.1%
    MAX_SHARES_PER_STOCK: int = 10000
    LOOKBACK_WINDOW: int = 20

    # PPO hyperparameters (defaults)
    PPO_LEARNING_RATE: float = 3e-4
    PPO_N_STEPS: int = 2048
    PPO_BATCH_SIZE: int = 64
    PPO_N_EPOCHS: int = 10
    PPO_GAMMA: float = 0.99
    PPO_GAE_LAMBDA: float = 0.95
    PPO_CLIP_RANGE: float = 0.2
    PPO_ENT_COEF: float = 0.01
    PPO_VF_COEF: float = 0.5
    PPO_MAX_GRAD_NORM: float = 0.5

    # Training
    TOTAL_TIMESTEPS: int = 200_000
    EVAL_FREQ: int = 10_000
    N_EVAL_EPISODES: int = 5

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = "ppo_trading"

    # DagsHub
    DAGSHUB_USERNAME: str = os.getenv("DAGSHUB_USERNAME", "")
    DAGSHUB_REPO: str = os.getenv("DAGSHUB_REPO", "")
    DAGSHUB_TOKEN: str = os.getenv("DAGSHUB_TOKEN", "")

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @classmethod
    def get_split_dates(cls) -> Tuple[str, str, str]:
        """Return train/val/test split boundaries."""
        return cls.START_DATE, cls.VALIDATION_SPLIT_DATE, cls.TEST_SPLIT_DATE

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all necessary directories."""
        for dir_path in [
            cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.FEATURED_DATA_DIR,
            cls.MODELS_DIR / "trained", cls.MODELS_DIR / "best", cls.MODELS_DIR / "registry",
            cls.MLRUNS_DIR, cls.ARTIFACTS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)