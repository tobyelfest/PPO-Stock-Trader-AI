import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from stable_baselines3 import PPO

# Tambahkan root project ke path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modul internal
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

# ========================
# Konfigurasi Halaman
# ========================
st.set_page_config(page_title="AI Trading Bot - BBCA", layout="wide")
st.title("🤖 AI Stock Trading Dashboard (PPO)")
st.markdown("---")

# ========================
# Sidebar Info
# ========================
st.sidebar.header("📊 Informasi Bot")
st.sidebar.info(f"**Target Saham:** {Config.STOCKS}")
st.sidebar.info(f"**Periode Training:** {Config.START_DATE} → {Config.VALIDATION_SPLIT_DATE}")
st.sidebar.info(f"**Periode Testing:** {Config.VALIDATION_SPLIT_DATE} → {Config.END_DATE}")
st.sidebar.info(f"**Modal Awal:** Rp {Config.INITIAL_CAPITAL:,.0f}")
st.sidebar.markdown("---")
st.sidebar.warning("Pastikan file **`trained_ppo_model.zip**` ada di direktori yang sama.")

MODEL_PATH = "trained_ppo_model.zip"

# ========================
# Tombol Eksekusi
# ========================
if st.button("🚀 Jalankan Backtest Sekarang", type="primary"):
    try:
        with st.status("⏳ Memproses data & menjalankan AI...", expanded=True) as status:
            # ------------------------------
            # 1. Load data
            # ------------------------------
            st.write("📥 Mengunduh/membaca data historis...")
            loader = DataLoader()
            df_raw = loader.download_all()   # returns MultiIndex (Date, Ticker)

            # Untuk single stock, kita ambil ticker pertama
            ticker = Config.STOCKS[0]
            if isinstance(df_raw.columns, pd.MultiIndex):
                # df_raw memiliki kolom (Ticker, OHLCV)
                df = df_raw.xs(ticker, level=0, axis=1).copy()
            else:
                df = df_raw.copy()

            # Pastikan kolom yang dibutuhkan ada
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Kolom tidak lengkap. Ditemukan: {df.columns.tolist()}")
                st.stop()

            # Reset index jika Date bukan index (jadikan index datetime)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)

            # ------------------------------
            # 2. Hitung indikator teknikal
            # ------------------------------
            st.write("📊 Menghitung indikator teknikal...")
            df = TechnicalIndicators.compute_all(df)
            df = MarketRegime.add_regime_features(df)   # tambahan regime market
            df = df.dropna()

            # ------------------------------
            # 3. Split data train / test
            # ------------------------------
            st.write("✂️ Memisahkan data training & testing...")
            splitter = DataSplitter()
            train_end = Config.VALIDATION_SPLIT_DATE
            test_end = Config.END_DATE
            train_df, val_df, test_df = splitter.split_by_date(df, train_end, test_end)

            if test_df.empty:
                st.error(f"Data testing kosong. Cek tanggal VALIDATION_SPLIT_DATE: {Config.VALIDATION_SPLIT_DATE}")
                st.stop()

            # ------------------------------
            # 4. Normalisasi fitur (opsional)
            # ------------------------------
            # Kita akan gunakan fitur yang ada di Config.INDICATORS + 'Close'
            feature_cols = list(Config.INDICATORS) + ['Close', 'Volume', 'regime', 'volatility_20d']
            # Hanya yang tersedia di dataframe
            feature_cols = [c for c in feature_cols if c in test_df.columns]

            preprocessor = DataPreprocessor(features=feature_cols, lookback=Config.LOOKBACK_WINDOW)
            preprocessor.fit(train_df[feature_cols])
            test_df_norm = preprocessor.transform(test_df[feature_cols])
            # Gabungkan kembali dengan kolom harga asli (dibutuhkan environment)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                test_df_norm[col] = test_df[col]

            # ------------------------------
            # 5. Load model PPO
            # ------------------------------
            st.write("🧠 Memuat model PPO yang sudah dilatih...")
            if not os.path.exists(MODEL_PATH):
                st.error(f"File model `{MODEL_PATH}` tidak ditemukan. Letakkan di folder yang sama dengan app.py")
                st.stop()
            model = PPO.load(MODEL_PATH)

            # ------------------------------
            # 6. Setup environment dengan risk management
            # ------------------------------
            st.write("🏃 Menjalankan simulasi trading...")
            # Gunakan position sizer, stop loss, take profit
            position_sizer = FixedFractionSizer()
            stop_loss = TrailingStopLoss(trail_percent=0.03)
            take_profit = FixedTakeProfit(ratio=2.0)

            env = TradingEnv(
                df=test_df_norm,
                stocks=[ticker],
                initial_capital=Config.INITIAL_CAPITAL,
                transaction_cost_pct=Config.TRANSACTION_COST_PCT,
                lookback_window=Config.LOOKBACK_WINDOW,
                position_sizer=position_sizer,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            # ------------------------------
            # 7. Run episode
            # ------------------------------
            obs, info = env.reset()
            done = False
            portfolio_values = [info["portfolio_value"]]
            actions_taken = []

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                portfolio_values.append(info["portfolio_value"])
                actions_taken.append(action)
                done = terminated or truncated

            status.update(label="✅ Backtest selesai!", state="complete", expanded=False)

        # ================================
        # 8. Tampilkan Hasil
        # ================================
        st.success("🎉 **Backtest berhasil dijalankan!**")

        # Hitung metrik
        equity_curve = pd.Series(portfolio_values)
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0] * 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        max_dd = PortfolioRisk.max_drawdown(equity_curve) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modal Akhir", f"Rp {equity_curve.iloc[-1]:,.0f}", delta=f"{total_return:.2f}%")
        col2.metric("Total Return", f"{total_return:.2f}%")
        col3.metric("Sharpe Ratio (Annual)", f"{sharpe:.2f}")
        col4.metric("Max Drawdown", f"{max_dd:.2f}%")

        # Grafik equity curve
        st.subheader("📈 Kurva Pertumbuhan Portfolio")
        st.line_chart(equity_curve)

        # Distribusi aksi (opsional)
        if len(actions_taken) > 0:
            actions_df = pd.DataFrame(actions_taken, columns=[f"Aksi_{ticker}"])
            st.subheader("🎮 Distribusi Aksi (0=Hold, 1=Buy, 2=Sell)")
            st.bar_chart(actions_df.value_counts())

        # Tabel ringkasan
        with st.expander("📋 Detail Data Testing"):
            st.write(f"**Periode:** {test_df.index[0].date()} → {test_df.index[-1].date()}")
            st.write(f"**Jumlah hari trading:** {len(test_df)}")
            st.write(f"**Jumlah step environment:** {len(portfolio_values)-1}")

        # Opsi download hasil
        results = pd.DataFrame({
            "step": range(len(portfolio_values)),
            "portfolio_value": portfolio_values
        })
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Hasil (CSV)", csv, "backtest_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        st.info("Periksa kembali struktur folder, file model, dan koneksi internet untuk download data.")
