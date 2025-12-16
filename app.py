import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly.express as px


# 1. KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Dashboard Prediksi TPaK - Kelompok 9",
    page_icon="📈",
    layout="wide"
)


# 2. CSS KUSTOM (UI DASHBOARD)

st.markdown("""
<style>
/* GLOBAL */
.stApp {
    background: radial-gradient(circle at top, #1f3c4d, #0f2027);
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}

/* HEADINGS */
h1 { font-size: 2.6rem; font-weight: 800; }
h2 { font-size: 1.8rem; font-weight: 700; }
h3 { font-size: 1.3rem; font-weight: 600; }

/* CARD */
.card {
    background: linear-gradient(160deg, rgba(255,255,255,0.14), rgba(255,255,255,0.04));
    padding: 26px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    margin-bottom: 26px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.6);
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg, #ff512f, #f09819);
    color: #ffffff;
    border-radius: 999px;
    padding: 0.75em 2.2em;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    width: 100%;
}
.stButton > button:hover {
    filter: brightness(1.1);
    transform: scale(1.02);
}

/* METRIC */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.18);
    padding: 18px;
    border-radius: 16px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffcc70;
}
</style>
""", unsafe_allow_html=True)


# 3. LOAD MODEL & SCALER

@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("model_regresi.pkl")
        return model, scaler, True
    except Exception:
        scaler = StandardScaler()
        cols = ['jumlah_penduduk', 'jumlah_penduduk_miskin', 'lama_sekolah']
        dummy_X = pd.DataFrame([[1500.0, 50.0, 9.0]], columns=cols)
        scaler.fit(dummy_X)
        model = LinearRegression()
        model.fit(scaler.transform(dummy_X), np.array([65.0]))
        return model, scaler, False

model_tpak, scaler_tpak, loaded = load_assets()

if loaded:
    st.sidebar.success("✅ Model & Scaler Berhasil Dimuat")
else:
    st.sidebar.warning("⚠️ File .pkl tidak ditemukan (Model Simulasi)")


# 4. HEADER UTAMA

st.markdown("""
<div class="card">
    <h1>📈 Dashboard Prediksi Tingkat Partisipasi Angkatan Kerja</h1>
    <p style="font-size:1.1rem; opacity:0.85;">
        Analisis TPaK Provinsi Jawa Barat menggunakan <b>Regresi Linear</b><br>
        <b>Kelompok 9</b> | Sumber Data: BPS Jawa Barat
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📊 Evaluasi Model", "🎯 Kalkulator Prediksi"])


# TAB 1 – BERANDA

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🏠 Tentang Proyek")
    st.write("""
    Aplikasi ini bertujuan untuk mengestimasi **Tingkat Partisipasi Angkatan Kerja (TPaK)** 
    berdasarkan indikator kependudukan dan pendidikan di Provinsi Jawa Barat.

    Variabel yang digunakan dalam pemodelan ini meliputi:
    - **Jumlah Penduduk** (dalam ribu orang)
    - **Jumlah Penduduk Miskin** (dalam ribu jiwa)
    - **Rata-rata Lama Sekolah** (dalam tahun)

    Model yang digunakan adalah **Regresi Linear**, yang bertujuan menangkap hubungan linier
    antara variabel independen terhadap TPaK.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# TAB 2 – EVALUASI MODEL

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Evaluasi Performa Model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", "3.067")
    c2.metric("MSE", "14.614")
    c3.metric("RMSE", "3.822")
    c4.metric("R² Score", "0.157")

    st.info("""
    📌 **Interpretasi**  
    Nilai R² sebesar 0,157 menunjukkan bahwa model memiliki keterbatasan dalam
    menjelaskan variasi TPaK. Hal ini mengindikasikan adanya faktor lain di luar
    variabel penelitian yang memengaruhi tingkat partisipasi angkatan kerja.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# TAB 3 – PREDIKSI

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🎯 Input Data Wilayah")

    col1, col2 = st.columns(2)
    with col1:
        jml_penduduk = st.number_input(
            "1. Jumlah Penduduk (Ribu Orang)",
            min_value=1.0, value=1500.0, step=10.0,
            help="Contoh: 1500 berarti 1.500.000 jiwa"
        )
        jml_miskin = st.number_input(
            "2. Jumlah Penduduk Miskin (Ribu Jiwa)",
            min_value=0.1, value=150.0, step=1.0,
            help="Contoh: 451.3 berarti 451.300 jiwa"
        )
    with col2:
        lama_sekolah = st.number_input(
            "3. Rata-rata Lama Sekolah (Tahun)",
            min_value=1.0, max_value=20.0, value=9.0, step=0.1
        )
        st.info("💡 Dataset menggunakan satuan **ribuan**, bukan jutaan.")

    if st.button("🔮 Hitung Estimasi TPaK"):
        cols = ['jumlah_penduduk', 'jumlah_penduduk_miskin', 'lama_sekolah']
        input_df = pd.DataFrame([[jml_penduduk, jml_miskin, lama_sekolah]], columns=cols)
        input_scaled = scaler_tpak.transform(input_df)
        hasil = model_tpak.predict(input_scaled)[0]

        st.markdown(f"""
        <div class="card" style="
            text-align:center;
            border: 3px solid #ffcc70;
            background: linear-gradient(160deg, rgba(255,204,112,0.18), rgba(255,255,255,0.05));
        ">
            <h2 style="letter-spacing:2px;">HASIL ESTIMASI</h2>
            <h1 style="color:#ffcc70; font-size:4.2rem; margin:10px 0;">
                {hasil:.2f} %
            </h1>
            <p>Perkiraan Tingkat Partisipasi Angkatan Kerja</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# SIDEBAR

st.sidebar.markdown("### 📘 Informasi Model")
st.sidebar.write("Metode: Regresi Linear")
st.sidebar.write("Variabel: Kependudukan & Pendidikan")
st.sidebar.write("Kelompok: 9")
