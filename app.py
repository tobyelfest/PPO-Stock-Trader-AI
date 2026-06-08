import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data.loader import DataLoader
from data.preprocessing import DataPreprocessor
from data.splitter import DataSplitter
from features.indicators import TechnicalIndicators
from features.market_regime import MarketRegime
from environment.trading_env import TradingEnv
from risk_management.position_sizing import FixedFractionSizer
from risk_management.stop_loss import TrailingStopLoss
from risk_management.take_profit import FixedTakeProfit
from risk_management.portfolio_risk import PortfolioRisk

st.set_page_config(page_title="AI Trading Bot - BBCA", layout="wide")
st.title("🤖 AI Stock Trading Dashboard (PPO)")
st.markdown("---")

st.sidebar.header("📊 Informasi Bot")
st.sidebar.info(f"**Target Saham:** {Config.STOCKS[0]}")
st.sidebar.info(f"**Periode Testing:** {Config.VALIDATION_SPLIT_DATE} → {Config.END_DATE}")
st.sidebar.info(f"**Modal Awal:** Rp {Config.INITIAL_CAPITAL:,.0f}")
st.sidebar.markdown("---")

MODEL_PATH = "trained_ppo_model.zip"

if st.button("🚀 Jalankan Backtest Sekarang", type="primary"):
    try:
        with st.status("⏳ Memproses data...", expanded=True) as status:
            # 1. Load data
            st.write("📥 Mengunduh data...")
            loader = DataLoader()
            df_raw = loader.download_all()
            
            ticker = Config.STOCKS[0]
            if isinstance(df_raw.columns, pd.MultiIndex):
                df = df_raw.xs(ticker, level=0, axis=1).copy()
            else:
                df = df_raw.copy()
            
            # Pastikan kolom yang diperlukan ada
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required):
                st.error(f"Kolom tidak lengkap: {df.columns.tolist()}")
                st.stop()
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # 2. Indikator
            st.write("📊 Menghitung indikator...")
            df = TechnicalIndicators.compute_all(df)
            df = MarketRegime.add_regime_features(df)
            df = df.dropna()
            
            # 3. Split data
            st.write("✂️ Split data...")
            splitter = DataSplitter()
            train_df, val_df, test_df = splitter.split_by_date(df, Config.VALIDATION_SPLIT_DATE, Config.END_DATE)
            
            if test_df.empty:
                st.error("Data testing kosong!")
                st.stop()
            
            # 4. Normalisasi fitur (opsional)
            feature_cols = [c for c in Config.INDICATORS if c in test_df.columns] + ['Close', 'Volume', 'regime', 'volatility_20d']
            feature_cols = list(set([c for c in feature_cols if c in test_df.columns]))
            preprocessor = DataPreprocessor(features=feature_cols, lookback=Config.LOOKBACK_WINDOW)
            preprocessor.fit(train_df[feature_cols])
            test_df_norm = preprocessor.transform(test_df[feature_cols])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                test_df_norm[col] = test_df[col]
            
            # 5. Load model
            st.write("🧠 Load model PPO...")
            if not os.path.exists(MODEL_PATH):
                st.error(f"Model {MODEL_PATH} tidak ditemukan!")
                st.stop()
            model = PPO.load(MODEL_PATH)
            
            # 6. Setup environment (single stock)
            st.write("🏃 Simulasi trading...")
            env = TradingEnv(
                df=test_df_norm,
                initial_capital=Config.INITIAL_CAPITAL,
                transaction_cost_pct=Config.TRANSACTION_COST_PCT,
                lookback_window=Config.LOOKBACK_WINDOW,
                position_sizer=FixedFractionSizer(),
                stop_loss=TrailingStopLoss(trail_percent=0.03),
                take_profit=FixedTakeProfit(ratio=2.0)
            )
            
            # 7. Run episode
            obs, info = env.reset()
            done = False
            portfolio_values = [info["portfolio_value"]]
            actions = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                portfolio_values.append(info["portfolio_value"])
                actions.append(action)
                done = terminated or truncated
            
            status.update(label="✅ Selesai!", state="complete", expanded=False)
        
        # 8. Tampilkan hasil
        st.success("Backtest berhasil!")
        equity = pd.Series(portfolio_values)
        ret = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100
        returns = equity.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        max_dd = PortfolioRisk.max_drawdown(equity) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modal Akhir", f"Rp {equity.iloc[-1]:,.0f}", delta=f"{ret:.2f}%")
        col2.metric("Total Return", f"{ret:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        st.subheader("Kurva Portfolio")
        st.line_chart(equity)
        
        # Distribusi aksi
        action_counts = pd.Series(actions).value_counts()
        st.subheader("Distribusi Aksi (0=Hold, 1=Buy, 2=Sell)")
        st.bar_chart(action_counts)
        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        st.info("Periksa kembali struktur folder dan file model.")
