import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
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
st.sidebar.info("**Target Saham:** BBCA.JK")
st.sidebar.info(f"**Modal Awal:** Rp {Config.INITIAL_CAPITAL:,.0f}")
st.sidebar.markdown("---")

MODEL_PATH = "trained_ppo_model.zip"
CSV_PATH = "BBCA.JK.csv"

if st.button("🚀 Jalankan Backtest Sekarang", type="primary"):
    try:
        with st.status("⏳ Memproses data...", expanded=True) as status:
            # 1. Baca CSV
            st.write("📥 Membaca data BBCA.JK.csv...")
            if not os.path.exists(CSV_PATH):
                st.error(f"File {CSV_PATH} tidak ditemukan di repository!")
                st.stop()
            df = pd.read_csv(CSV_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required):
                st.error(f"Kolom tidak lengkap: {df.columns.tolist()}")
                st.stop()
            
            # 2. Hitung indikator
            st.write("📊 Menghitung indikator teknikal...")
            df = TechnicalIndicators.compute_all(df)
            df = MarketRegime.add_regime_features(df)
            df = df.dropna()
            if len(df) < Config.LOOKBACK_WINDOW + 10:
                st.error(f"Data terlalu sedikit ({len(df)} baris) setelah preprocessing. Kurangi LOOKBACK_WINDOW atau gunakan data lebih panjang.")
                st.stop()
            # 3. Siapkan data testing (semua data setelah preprocessing)
            test_df = df.copy()
            if test_df.empty:
                st.error("Data kosong setelah preprocessing!")
                st.stop()
            
            # 4. Load model
            st.write("🧠 Memuat model PPO...")
            if not os.path.exists(MODEL_PATH):
                st.error(f"Model {MODEL_PATH} tidak ditemukan!")
                st.stop()
            model = PPO.load(MODEL_PATH)
            
            # 5. Environment
            st.write("🏃 Simulasi trading...")
            env = TradingEnv(
                df=test_df,
                initial_capital=Config.INITIAL_CAPITAL,
                transaction_cost_pct=Config.TRANSACTION_COST_PCT,
                lookback_window=Config.LOOKBACK_WINDOW,
                position_sizer=FixedFractionSizer(),
                stop_loss=TrailingStopLoss(trail_percent=0.03),
                take_profit=FixedTakeProfit(ratio=2.0)
            )
            
            # 6. Run episode
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
        
        # Tampilkan hasil
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
        
        if actions:
            st.subheader("Distribusi Aksi")
            st.bar_chart(pd.Series(actions).value_counts().sort_index())
        
        # Download
        results = pd.DataFrame({"step": range(len(portfolio_values)), "portfolio_value": portfolio_values})
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download CSV", csv, "backtest_results.csv", "text/csv")
        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        st.info("Periksa kembali file model dan pastikan semua file sudah di-upload.")
