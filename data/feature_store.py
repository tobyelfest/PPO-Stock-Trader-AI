import pandas as pd
from pathlib import Path
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Save and load processed feature DataFrames."""

    def __init__(self, name: str = "featured_data.parquet"):
        self.path = Config.FEATURED_DATA_DIR / name

    def save(self, df: pd.DataFrame) -> None:
        """Save DataFrame with features."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.path)
        logger.info(f"Saved featured data to {self.path}")

    def load(self) -> pd.DataFrame:
        """Load featured data."""
        if not self.path.exists():
            raise FileNotFoundError(f"Feature store not found: {self.path}")
        return pd.read_parquet(self.path)