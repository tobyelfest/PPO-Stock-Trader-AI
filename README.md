# 🤖 PPO Stock Trading AI Bot (BBCA.JK)

Project ini adalah sistem backtesting otomatis untuk saham **BBCA** menggunakan algoritma **Reinforcement Learning (PPO)**. Dashboard ini dibangun dengan Streamlit untuk memvisualisasikan performa AI dalam mengambil keputusan beli dan jual.

## 🚀 Fitur Utama
* **Artificial Intelligence**: Menggunakan model PPO (Proximal Policy Optimization) yang sudah terlatih.
* **Real-time Data**: Menarik data historis langsung dari Yahoo Finance API.
* **Technical Indicators**: Dilengkapi dengan perhitungan SMA, RSI, MACD, dan indikator teknikal lainnya.
* **Interactive Backtest**: Visualisasi grafik portfolio secara interaktif melalui dashboard web.

## 🛠️ Tech Stack
* **Language**: Python
* **AI Framework**: Stable-Baselines3
* **Dashboard**: Streamlit
* **Data Science**: Pandas, NumPy, yfinance
* **Environment**: Custom Gymnasium Trading Environment

## 📁 Struktur Folder
- `app.py`: File utama dashboard web.
- `config/`: Konfigurasi parameter trading dan saham.
- `data/`: Script untuk penarikan data historis.
- `environment/`: Logika pasar simulasi (market rules).
- `features/`: Script perhitungan indikator teknis.
- `trained_ppo_model.zip`: Model AI yang sudah dilatih (The Brain).

## 📊 Cara Menjalankan
1. Pastikan semua library terinstall melalui `pip install -r requirements.txt`.
2. Jalankan perintah `streamlit run app.py` di terminal Anda.
3. Klik tombol **"Jalankan Backtest"** pada dashboard.

---
*Dibuat untuk tujuan edukasi dan eksperimen Reinforcement Learning pada pasar modal Indonesia.*
