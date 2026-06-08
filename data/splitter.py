import pandas as pd

class DataSplitter:
    """Time-based train/validation/test split."""

    @staticmethod
    def split_by_date(df: pd.DataFrame,
                      train_end: str,
                      val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split based on date indices.

        Args:
            df: DataFrame with DateTimeIndex
            train_end: inclusive end date for training (YYYY-MM-DD)
            val_end: inclusive end date for validation

        Returns:
            (train_df, val_df, test_df)
        """
        train = df[df.index <= train_end]
        val = df[(df.index > train_end) & (df.index <= val_end)]
        test = df[df.index > val_end]

        return train, val, test