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

st.set_page_config(page_title="AI Trading Bot BBCA", layout="wide")
st.title("🤖 AI Stock Trading Dashboard")
st.markdown("---")

MODEL_NAME = "trained_ppo_model"

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
            
            # --- AGGRESSIVE FIX UNTUK ERROR 'TICKER' ---
            df = df_raw.copy()
            
            # A. Buang MultiIndex di Kolom
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # B. Paksa Reset Index (Buang semua level Ticker/Date yang bikin error)
            df = df.reset_index()
            
            # C. Cari kolom yang mengandung tanggal dan jadikan Index murni
            # Biasanya namanya 'Date' atau 'index' setelah di reset_index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'index' in df.columns:
                df['index'] = pd.to_datetime(df['index'])
                df.set_index('index', inplace=True)
                df.index.name = 'Date'
            
            # D. Buang kolom 'Ticker' jika dia nyasar jadi kolom biasa
            if 'Ticker' in df.columns:
                df = df.drop(columns=['Ticker'])
                
            # E. Pastikan index bersih dari zona waktu (Timezone)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # -------------------------------------------
            
            # 2. Proses Indikator
            st.write("📊 Menghitung indikator teknikal...")
            df = TechnicalIndicators.compute_all(df)
            df = df.dropna()
            
            # 3. Filter Data
            test_start = pd.to_datetime(Config.VALIDATION_SPLIT_DATE)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if test_start.tz is not None:
                test_start = test_start.tz_localize(None)
                
            test_df = df[df.index >= test_start].copy()
            
            if test_df.empty:
                st.error("Data testing kosong!")
                st.stop()
            
            # 4. Load Model
            st.write("🧠 Memuat otak AI (PPO)...")
            model = PPO.load(MODEL_NAME)
            
            # 5. Simulasi
            st.write("🏃 AI sedang bertransaksi...")
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

        st.success("✅ Backtest Berhasil!")
        
        c1, c2, c3 = st.columns(3)
        initial_val = portfolio_values[0]
        final_val = portfolio_values[-1]
        profit_pct = ((final_val - initial_val) / initial_val) * 100
        
        c1.metric("Modal Awal", f"Rp {initial_val:,.0f}")
        c2.metric("Saldo Akhir", f"Rp {final_val:,.0f}")
        c3.metric("Profit/Loss", f"{profit_pct:.2f}%", delta=f"{profit_pct:.2f}%")
        
        st.subheader("Grafik Pertumbuhan Portfolio")
        st.line_chart(portfolio_values)
        
    except Exception as e:
        st.error(f"Waduh, ada masalah teknis: {e}")
