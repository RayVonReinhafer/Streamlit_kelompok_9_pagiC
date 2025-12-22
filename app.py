import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ======================================================
# 1. KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Dashboard Prediksi TPaK - Kelompok 9",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# 2. CSS KUSTOM
# ======================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #1f3c4d, #0f2027);
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}
h1 { font-size: 2.6rem; font-weight: 800; }
.card {
    background: linear-gradient(160deg, rgba(255,255,255,0.14), rgba(255,255,255,0.04));
    padding: 26px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    margin-bottom: 26px;
}
.stButton > button {
    background: linear-gradient(90deg, #ff512f, #f09819);
    color: #ffffff;
    border-radius: 999px;
    padding: 0.75em 2.2em;
    font-weight: 700;
    border: none;
    width: 100%;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.18);
    padding: 18px;
    border-radius: 16px;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD DATA EDA
# ======================================================
@st.cache_data
def load_eda_data():
    df_tpak = pd.read_csv('tpak.csv', sep=';', decimal=',')
    df_penduduk = pd.read_csv('penduduk.csv', sep=';', decimal=',')
    df_miskin = pd.read_csv('penduduk_miskin.csv', sep=';', decimal=',')
    df_sekolah = pd.read_csv('avg.csv', sep=';', decimal=',')
    df_umr = pd.read_csv('umr.csv', sep=';', decimal=',')

    df = pd.merge(
        df_tpak,
        df_penduduk[['nama_kabupaten_kota','tahun','jumlah_penduduk']],
        on=['nama_kabupaten_kota','tahun']
    )
    df = pd.merge(
        df,
        df_miskin[['nama_kabupaten_kota','tahun','jumlah_penduduk_miskin']],
        on=['nama_kabupaten_kota','tahun']
    )
    df = pd.merge(
        df,
        df_sekolah[['nama_kabupaten_kota','tahun','lama_sekolah']],
        on=['nama_kabupaten_kota','tahun']
    )
    df = pd.merge(
        df,
        df_umr[['nama_kabupaten_kota','tahun','besaran_upah_minimum']],
        on=['nama_kabupaten_kota','tahun']
    )

    df = df[df['tahun'].between(2022, 2024)]
    return df

# ======================================================
# 4. LOAD MODEL & SCALER
# ======================================================
@st.cache_resource
def load_ml_assets():
    model = joblib.load("model_regresi.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler, True

df_eda = load_eda_data()
model_tpak, scaler_tpak, ml_loaded = load_ml_assets()

# ======================================================
# 5. HEADER
# ======================================================
st.markdown("""
<div class="card">
    <h1>📈 Dashboard Analisis & Prediksi TPaK</h1>
    <p style="opacity:0.85;">Provinsi Jawa Barat | Periode 2022–2024 | Regresi Linear</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Beranda",
    "📊 Visualisasi EDA",
    "📉 Evaluasi Model",
    "🎯 Kalkulator Prediksi"
])

# ======================================================
# TAB 1 — BERANDA
# ======================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    Dashboard ini menganalisis faktor-faktor yang mempengaruhi
    **Tingkat Partisipasi Angkatan Kerja (TPaK)** di Provinsi Jawa Barat
    menggunakan pendekatan **Regresi Linear**.
    """)
    st.dataframe(df_eda.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# TAB 2 — EDA
# ======================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        avg_tpak = df_eda.groupby('tahun')['tingkat_partisipasi_angkatan_kerja'].mean().reset_index()
        fig = px.line(avg_tpak, x='tahun', y='tingkat_partisipasi_angkatan_kerja',
                      markers=True, title="Rata-rata TPaK per Tahun")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        corr = df_eda[
            ['tingkat_partisipasi_angkatan_kerja',
             'jumlah_penduduk',
             'jumlah_penduduk_miskin',
             'lama_sekolah',
             'besaran_upah_minimum']
        ].corr()
        fig = px.imshow(corr, text_auto=True, title="Heatmap Korelasi")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# TAB 3 — EVALUASI
# ======================================================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", "2.82")
    c2.metric("MSE", "12.76")
    c3.metric("RMSE", "3.57")
    c4.metric("R² Score", "0.264")
    st.info("Model menggunakan data 2022–2024 dengan penambahan variabel UMR.")
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# TAB 4 — PREDIKSI (FIXED)
# ======================================================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        jml_penduduk = st.number_input("Jumlah Penduduk (Ribu Orang)", 1.0, 10000.0, 1500.0)
        jml_miskin = st.number_input("Penduduk Miskin (Ribu Jiwa)", 0.1, 2000.0, 200.0)

    with col2:
        lama_sekolah = st.number_input("Rata-rata Lama Sekolah (Tahun)", 1.0, 20.0, 9.0)
        umr = st.number_input("UMR (Juta Rupiah)", 1.0, 10.0, 4.0)


    if st.button("🔮 Prediksi TPaK"):
        X = pd.DataFrame([[
            jml_penduduk,
            jml_miskin,
            lama_sekolah,
            umr * 1_000_000
        ]], columns=[
            'jumlah_penduduk',
            'jumlah_penduduk_miskin',
            'lama_sekolah',
            'besaran_upah_minimum'
        ])

        X_scaled = scaler_tpak.transform(X)
        hasil = model_tpak.predict(X_scaled)[0]

        # 🔒 BATASI NILAI AGAR REALISTIS
        hasil = max(0, min(100, hasil))

        st.markdown(f"""
        <div class="card" style="text-align:center; border:3px solid #ffcc70;">
            <h2>HASIL ESTIMASI TPaK</h2>
            <h1 style="color:#ffcc70; font-size:4.5rem;">{hasil:.2f} %</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📘 Informasi")
st.sidebar.info("""
Dashboard ini dikembangkan oleh Kelompok 9  
Analisis Regresi Linear TPaK  
Provinsi Jawa Barat (2022–2024)
""")
