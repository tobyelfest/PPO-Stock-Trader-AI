import hashlib
import json
from pathlib import Path
from datetime import datetime
from config.config import Config

class DataVersion:
    """Simple versioning using file hash and timestamp."""

    @staticmethod
    def compute_hash(df) -> str:
        """Compute SHA256 hash of a DataFrame's content."""
        # Convert to CSV string without index
        csv_str = df.to_csv(index=False)
        return hashlib.sha256(csv_str.encode()).hexdigest()

    @staticmethod
    def save_version_metadata(df, name: str = "latest") -> None:
        """Save version info to JSON."""
        version_info = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "hash": DataVersion.compute_hash(df),
            "shape": list(df.shape)
        }
        version_file = Config.FEATURED_DATA_DIR / "version.json"
        with open(version_file, "w") as f:
            json.dump(version_info, f, indent=2)