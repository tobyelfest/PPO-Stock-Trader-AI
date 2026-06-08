import streamlit as st
import pandas as pd
import os
import sys
from stable_baselines3 import PPO

# Menghubungkan folder-folder project lo ke sistem
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.trading_env import TradingEnv
from data.loader import DataLoader
from features.indicators import TechnicalIndicators
from config.config import Config

# Konfigurasi Halaman
st.set_page_config(page_title="AI Trading Bot BBCA", layout="wide")

st.title("🤖 AI Stock Trading Dashboard")
st.markdown("---")

# Nama file model lo di GitHub (tanpa ekstensi .zip)
MODEL_NAME = "trained_ppo_model"

# Sidebar untuk informasi
st.sidebar.header("Informasi Bot")
st.sidebar.info(f"Target Saham: {Config.STOCKS}")
st.sidebar.write(f"Mulai Testing: {Config.VALIDATION_SPLIT_DATE}")

if st.button("🚀 Jalankan Backtest Sekarang"):
    try:
        with st.status("Sedang bekerja...", expanded=True) as status:
            # 1. Download Data
            st.write("📥 Mengambil data historis...")
            loader = DataLoader()
            df_raw = loader.download_all()
            
            # --- SUPER FIX: PERBAIKAN FORMAT INDEKS & KOLOM DATA YFINANCE ---
            df = df_raw.copy()
            
            # A. Ratakan Kolom yang Berlapis (MultiIndex Columns)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # B. Ratakan Indeks Baris jika Terbaca sebagai Tuple atau MultiIndex
            if isinstance(df.index, pd.MultiIndex):
                df.index = df.index.get_level_values(0)
            elif len(df.index) > 0 and isinstance(df.index[0], tuple):
                df.index = pd.to_datetime([x[0] for x in df.index])
            
            # C. Paksa Indeks Menjadi Datetime Murni & Bersihkan Nama Indeks
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            # -----------------------------------------------------------------
            
            # Step 2: Proses Indikator
            st.write("📊 Menghitung indikator teknikal...")
            df = TechnicalIndicators.compute_all(df)
            df = df.dropna()
            
            # Step 3: Filter Data untuk Testing
            test_start = pd.to_datetime(Config.VALIDATION_SPLIT_DATE)
            
            # Pastikan zona waktu (timezone) disamakan agar tidak bentrok saat filter tanggal
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if test_start.tz is not None:
                test_start = test_start.tz_localize(None)
                
            test_df = df[df.index >= test_start].copy()
            
            if test_df.empty:
                st.error("Data testing kosong. Cek 'VALIDATION_SPLIT_DATE' di config.py!")
                st.stop()
            
            # Step 4: Load Model AI
            st.write("🧠 Memuat otak AI (PPO)...")
            if not os.path.exists(f"{MODEL_NAME}.zip"):
                st.error(f"File {MODEL_NAME}.zip tidak ditemukan di repository!")
                st.stop()
            
            model = PPO.load(MODEL_NAME)
            
            # Step 5: Jalankan Simulasi
            st.write("🏃 AI sedang bertransaksi di pasar...")
            env = TradingEnv(test_df, stocks=Config.STOCKS)
            obs, info = env.reset()
            
            done = False
            portfolio_values = [info["portfolio_value"]]
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                portfolio_values.append(info["portfolio_value"])
                done = terminated or truncated
            
            status.update(label="Analisa Selesai!", state="complete", expanded=False)

        # Tampilkan Hasil
        st.success("✅ Backtest Berhasil!")
        
        # Ringkasan Angka
        c1, c2, c3 = st.columns(3)
        initial_val = portfolio_values[0]
        final_val = portfolio_values[-1]
        profit_pct = ((final_val - initial_val) / initial_val) * 100
        
        c1.metric("Modal Awal", f"Rp {initial_val:,.0f}")
        c2.metric("Saldo Akhir", f"Rp {final_val:,.0f}")
        c3.metric("Profit/Loss", f"{profit_pct:.2f}%", delta=f"{profit_pct:.2f}%")
        
        # Grafik
        st.subheader("Grafik Pertumbuhan Portfolio")
        st.line_chart(portfolio_values)
        
    except Exception as e:
        st.error(f"Waduh, ada masalah teknis: {e}")
        st.info("Pastikan folder 'data', 'config', 'environment', 'features', dan 'risk_management' sudah terupload dengan benar.")
