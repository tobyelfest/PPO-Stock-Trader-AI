import logging
import pandas as pd
import yfinance as yf
from config.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, stocks=Config.STOCKS, start_date=Config.START_DATE, end_date=Config.END_DATE):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.raw_dir = Config.RAW_DATA_DIR

    def download_all(self, force_refresh=False):
        all_data = []
        for ticker in self.stocks:
            filepath = self.raw_dir / f"{ticker}.csv"
            if filepath.exists():
                logger.info(f"Membaca CSV: {ticker}")
                df = pd.read_csv(filepath)
                df.columns = [c.strip().capitalize() for c in df.columns]
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                logger.info(f"Downloading: {ticker}")
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

            df["Ticker"] = ticker
            all_data.append(df)

        # --- JURUS PENYEMBUH (Biar Ticker jadi Index) ---
        combined = pd.concat(all_data)
        combined = combined.set_index("Ticker", append=True).reorder_levels(["Date", "Ticker"])
        combined.sort_index(inplace=True)
        return combined