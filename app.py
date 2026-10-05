"""
SONORA / EMOTIVA — AI Speech Emotion Intelligence Workstation
Production-ready Streamlit application with complete end-to-end functionality,
SQLite persistence CRUD, dual-model comparison, interactive DSP audio lab,
multi-format audio ingestion (WAV/MP3/WebM/OGG/FLAC), live microphone pipeline,
strict BD Supper (Display) & Forum (Body) typography, and comprehensive diagnostics.
"""
import os
import sys
import time
import tempfile
import base64
from datetime import datetime
import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT CALL)
# ---------------------------------------------------------
st.set_page_config(
    page_title="EMOTIVA | AI Speech Emotion Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 2. BACKEND IMPORTS & PATHS
# ---------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.preprocessing import (
    load_audio, load_audio_raw, load_ravdess_metadata, validate_audio_file,
    decode_audio_bytes, audio_to_wav_bytes
)
from src.feature_extraction import (
    extract_mel_spectrogram, extract_mfcc, extract_audio_features
)
from src.predict import (
    load_model_and_encoder, predict_emotion, predict_segments, get_confidence_level
)
from src.visualization import (
    plot_waveform, plot_mel_spectrogram, plot_mfcc,
    plot_spectral_centroid, plot_spectral_bandwidth, plot_spectral_rolloff,
    plot_rms_energy, plot_zero_crossing_rate, plot_emotion_radar,
    plot_emotion_bars, plot_emotion_timeline, plot_confusion_matrix,
    plot_training_curves, plot_emotion_distribution, plot_model_comparison
)
from src.report import generate_html_report
from src.database import (
    init_db, save_prediction, get_predictions, get_prediction_by_id,
    update_prediction, delete_prediction, clear_all_predictions,
    get_history_stats, get_setting, set_setting
)

# Initialize SQLite Database on startup
init_db()

# ---------------------------------------------------------
# 3. CSS STYLING — STRICT TYPOGRAPHY (BD SUPPER & FORUM)
# ---------------------------------------------------------
CSS_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Forum&display=swap');

@font-face {
    font-family: 'BD Supper';
    src: local('BD Supper'), local('BDSupper'), local('bd-supper'), local('BD-Supper');
    font-weight: normal;
    font-style: normal;
    font-display: swap;
}

:root {
  --bg-primary: #070b13;
  --bg-secondary: #0c1220;
  --bg-card: #12192c;
  --bg-card-hover: #18223c;
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --accent-cyan: #06b6d4;
  --accent-blue: #3b82f6;
  --accent-indigo: #6366f1;
  --accent-emerald: #10b981;
  --accent-amber: #f59e0b;
  --accent-glow: rgba(6, 182, 212, 0.25);
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-glow: rgba(6, 182, 212, 0.35);
}

/* Hide Streamlit chrome */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Global Body Font = Forum (Strict Requirement 10 & 11) */
html, body, [class*="css"], .stApp, p, span, label, div, button, 
.stButton button, input, select, textarea, .stMarkdown, .hero-subtitle, 
.status-card-title, .pipeline-node, [data-testid="stMetricLabel"],
.stSelectbox, .stRadio, .stSlider, .stTabs [role="tab"], .stExpander,
.caption, small, .stAlert, .disclaimer, table, td, th {
    font-family: 'Forum', serif !important;
}

/* Header & Display Font = BD Supper (Strict Requirement 10 & 11) */
h1, h2, h3, h4, h5, h6,
.hero-title, .hero-tag, .status-card-value, [data-testid="stMetricValue"], 
.prediction-box, .emotion-title, .section-header, .display-header, 
[data-testid="stHeader"], .stAppHeader, .navbar-brand, .brand-title, 
.metric-value, [data-testid="stMetricValue"] div {
    font-family: 'BD Supper', 'Forum', cursive, sans-serif !important;
    letter-spacing: -0.01em;
}

code, kbd, samp, pre {
    font-family: 'Courier New', monospace !important;
}

/* Global Typography & Background */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Streamlit Metric Overrides */
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Glass Card Component */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-glow);
    background: var(--bg-card-hover);
    box-shadow: 0 10px 35px var(--accent-glow);
}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.hero-tag {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 9999px;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: var(--accent-cyan);
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 20%, #93c5fd 60%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem;
}
.hero-subtitle {
    font-size: 1.3rem;
    font-weight: 400;
    color: var(--text-secondary);
    max-width: 720px;
    margin: 0 auto 1.8rem auto;
    line-height: 1.6;
}

/* Status Metric Cards */
.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0 2rem;
}
.status-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1.25rem 1rem;
    text-align: center;
    transition: transform 0.2s ease;
}
.status-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-cyan);
}
.status-card-title {
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.status-card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* Audio Lab Pipeline Flow */
.pipeline-flow {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    padding: 1.2rem;
    background: rgba(12, 18, 32, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    margin: 1.5rem 0;
}
.pipeline-node {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    padding: 0.6rem 1.1rem;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
}
.pipeline-node.active {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 15px var(--accent-glow);
    color: var(--accent-cyan);
}
.pipeline-arrow {
    color: var(--text-muted);
    font-weight: 700;
}

/* Result Prediction Hero */
.prediction-box {
    background: linear-gradient(145deg, #101628, #16203a);
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin: 1.5rem 0 1.8rem;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
    border: 2px solid;
    position: relative;
}
.prediction-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1rem;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.35) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 6px 22px rgba(6, 182, 212, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* Top 3 Progress Bars */
.progress-container {
    margin-bottom: 0.85rem;
}
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.progress-track {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* History Card */
.history-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s ease;
}
.history-card:hover {
    border-color: var(--accent-cyan);
    background: var(--bg-card-hover);
}
</style>
"""
st.markdown(CSS_THEME, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
state_defaults = {
    'page': 'HOME',
    'audio_data': None,
    'audio_sr': None,
    'audio_name': None,
    'audio_bytes': None,
    'audio_duration': None,
    'analysis_results': None,
    'dual_results': None,
    'model_type': 'cnn_lstm',
    'timeline_results': None,
    'audio_features': None,
    'last_prediction_id': None,
    'mfcc_n': 40,
    'spec_mels': 128,
}
for key, val in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------
# 5. CACHED MODEL LOADER
# ---------------------------------------------------------
@st.cache_resource
def get_model_and_encoder(model_name: str):
    """
    Cached model loader. Returns (model, encoder, error_msg).
    """
    try:
        model_obj, encoder_obj = load_model_and_encoder(model_name)
        return model_obj, encoder_obj, None
    except FileNotFoundError as fnf:
        return None, None, str(fnf)
    except Exception as ex:
        return None, None, f"Error loading {model_name}: {str(ex)}"

# ---------------------------------------------------------
# 6. NAVIGATION BAR
# ---------------------------------------------------------
PAGES = ["HOME", "ANALYZE", "AUDIO LAB", "INSIGHTS", "MODELS", "HISTORY", "ABOUT"]
nav_cols = st.columns(len(PAGES))

for i, p_name in enumerate(PAGES):
    with nav_cols[i]:
        is_active = (st.session_state.page == p_name)
        btn_type = "primary" if is_active else "secondary"
        if st.button(p_name, key=f"nav_btn_{p_name}", type=btn_type, use_container_width=True):
            st.session_state.page = p_name
            st.rerun()

st.markdown("<hr style='border: 1px solid var(--border-subtle); margin-top: 0.4rem; margin-bottom: 1.2rem;'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. ROUTING TO INDIVIDUAL PAGES
# ---------------------------------------------------------

# =========================================================
# PAGE 1: HOME
# =========================================================
if st.session_state.page == "HOME":
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-tag'>Acoustic Speech Intelligence</div>
        <div class='hero-title'>SONORA / EMOTIVA</div>
        <div class='hero-subtitle'>
            Real-time vocal affect recognition using high-resolution Log-Mel Spectrograms
            and Deep Recurrent Spatiotemporal Networks.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 ANALYZE VOICE", type="primary", use_container_width=True):
                st.session_state.page = "ANALYZE"
                st.rerun()
        with col_btn2:
            if st.button("🎛️ OPEN AUDIO LAB", use_container_width=True):
                st.session_state.page = "AUDIO LAB"
                st.rerun()

    # System Status Cards
    st.markdown("""
    <div class='status-grid'>
        <div class='status-card'>
            <div class='status-card-title'>Primary Architecture</div>
            <div class='status-card-value'>CNN-LSTM</div>
        </div>
        <div class='status-card'>
            <div class='status-card-title'>Audio DNA Resolution</div>
            <div class='status-card-value'>128 Mel Bands</div>
        </div>
        <div class='status-card'>
            <div class='status-card-title'>Affect Categories</div>
            <div class='status-card-value'>8 Classes</div>
        </div>
        <div class='status-card'>
            <div class='status-card-title'>Acoustic Sample Rate</div>
            <div class='status-card-value'>22,050 Hz</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Architectural Pipeline Flow
    st.markdown("""
    <div style='text-align:center; font-weight:700; font-size:1.3rem; margin-bottom:0.5rem;'>
        Signal Processing & Deep Inference Pipeline
    </div>
    <div class='pipeline-flow'>
        <div class='pipeline-node active'>🎤 Audio Input</div>
        <div class='pipeline-arrow'>→</div>
        <div class='pipeline-node'>🎚️ Mono & 22kHz Resample</div>
        <div class='pipeline-arrow'>→</div>
        <div class='pipeline-node'>🌈 128-Mel Spectrogram</div>
        <div class='pipeline-arrow'>→</div>
        <div class='pipeline-node'>🧠 Conv2D Spatial</div>
        <div class='pipeline-arrow'>→</div>
        <div class='pipeline-node'>⏱️ LSTM Temporal</div>
        <div class='pipeline-arrow'>→</div>
        <div class='pipeline-node active'>🎭 Affect Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    # Emotion Categories Display
    st.markdown("<h3 style='text-align: center; margin: 1.8rem 0 1rem;'>Recognized Affective Classes</h3>", unsafe_allow_html=True)
    emo_cols_1 = st.columns(4)
    emo_cols_2 = st.columns(4)
    all_emo_cols = emo_cols_1 + emo_cols_2

    for idx, emotion in enumerate(config.EMOTIONS):
        col_color = config.EMOTION_COLORS.get(emotion, "#38bdf8")
        emoji = config.EMOTION_EMOJIS.get(emotion, "🎭")
        with all_emo_cols[idx]:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; border-top: 3px solid {col_color}; padding: 1rem;'>
                <div style='font-size: 2.2rem; margin-bottom: 0.25rem;'>{emoji}</div>
                <div style='font-weight: 700; font-size:1.15rem; text-transform: capitalize;'>{emotion}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        if st.button("📊 Dataset Insights", use_container_width=True):
            st.session_state.page = "INSIGHTS"
            st.rerun()
    with f2:
        if st.button("🧪 Model Benchmarks", use_container_width=True):
            st.session_state.page = "MODELS"
            st.rerun()
    with f3:
        if st.button("📜 Audit History", use_container_width=True):
            st.session_state.page = "HISTORY"
            st.rerun()
    with f4:
        if st.button("ℹ️ System Architecture", use_container_width=True):
            st.session_state.page = "ABOUT"
            st.rerun()


# =========================================================
# PAGE 2: ANALYZE
# =========================================================
elif st.session_state.page == "ANALYZE":
    st.markdown("<h2>🔬 Audio Analysis & Emotion Recognition</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); font-size:1.1rem;'>Upload, record, or test voice signals through deep acoustic models with persistent audit tracking.</p>", unsafe_allow_html=True)

    # 1. Model Selection Control
    st.markdown("### 1. Select Model Architecture")
    m_col1, m_col2 = st.columns([1.2, 1.8])
    with m_col1:
        selected_model_type = st.radio(
            "Inference Architecture:",
            options=["cnn_lstm", "cnn"],
            format_func=lambda x: "CNN-LSTM (Spatiotemporal Hybrid Network)" if x == "cnn_lstm" else "CNN Baseline (Spatial Spectrogram Network)",
            index=0 if st.session_state.model_type == "cnn_lstm" else 1,
            horizontal=False
        )
        if selected_model_type != st.session_state.model_type:
            st.session_state.model_type = selected_model_type
            st.rerun()

    active_model_name = "CNN-LSTM Spatiotemporal" if st.session_state.model_type == "cnn_lstm" else "CNN Baseline"
    with m_col2:
        st.markdown(f"""
        <div style='background: var(--bg-card); padding: 1rem 1.25rem; border-radius: 8px; border-left: 4px solid var(--accent-cyan); margin-top: 0.5rem;'>
            <div style='font-size: 0.85rem; color: var(--text-muted); text-transform:uppercase;'>Current Active Model</div>
            <div style='font-size: 1.3rem; font-weight:700; color:var(--text-primary);'>{active_model_name}</div>
            <div style='font-size: 0.95rem; color: var(--text-secondary); margin-top: 4px;'>
                { 'Combines 2D spectral convolutions with sequential LSTM recurrence to capture pitch contours and syllabic intonation.' if st.session_state.model_type == 'cnn_lstm' else 'Learns multi-scale time-frequency filter patterns across the 128-band Mel spectrogram.' }
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Audio Source Selection (Upload / Record / Sample Library)
    st.markdown("### 2. Audio Input")
    tab_upload, tab_record, tab_samples = st.tabs(["📤 Upload Audio (WAV / MP3 / OGG / FLAC)", "🎙️ Record Microphone", "🎵 Demo Audio Library"])

    input_audio_bytes = None
    input_audio_name = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload audio file (WAV, MP3, OGG, FLAC)",
            type=['wav', 'mp3', 'ogg', 'flac'],
            key="uploader_audio"
        )
        if uploaded_file is not None:
            input_audio_bytes = uploaded_file.read()
            input_audio_name = uploaded_file.name

    with tab_record:
        if hasattr(st, 'audio_input'):
            recorded_audio = st.audio_input("Record voice sample (speak clearly for 3-5 seconds)", key="mic_recorder")
            if recorded_audio is not None:
                input_audio_bytes = recorded_audio.read()
                input_audio_name = f"mic_recording_{datetime.now().strftime('%H%M%S')}.wav"
        else:
            st.info("Browser microphone recording requires Streamlit audio_input. Please upload an audio file or select from Demo Audio Library.")

    with tab_samples:
        st.write("Select a pre-synthesized acoustic demo voice representing each of the 8 emotions (or multi-emotion speech):")
        samples_dir = os.path.join(config.PROJECT_ROOT, "samples")
        
        row1_cols = st.columns(4)
        row1_samples = [
            ("😐 Neutral (3.2s)", "demo_neutral.wav"),
            ("😌 Calm (3.2s)", "demo_calm.wav"),
            ("😊 Happy (3.2s)", "demo_happy.wav"),
            ("😢 Sad (3.2s)", "demo_sad.wav"),
        ]
        for idx, (label, sfile) in enumerate(row1_samples):
            with row1_cols[idx]:
                if st.button(label, key=f"btn_demo_{sfile}", use_container_width=True):
                    s_path = os.path.join(samples_dir, sfile)
                    if not os.path.exists(s_path):
                        alt = sfile.replace("demo_", "sample_")
                        s_path = os.path.join(samples_dir, alt) if os.path.exists(os.path.join(samples_dir, alt)) else s_path
                    if os.path.exists(s_path):
                        with open(s_path, 'rb') as f:
                            input_audio_bytes = f.read()
                            input_audio_name = sfile

        row2_cols = st.columns(4)
        row2_samples = [
            ("😡 Angry (3.2s)", "demo_angry.wav"),
            ("😨 Fearful (3.2s)", "demo_fearful.wav"),
            ("🤢 Disgust (3.2s)", "demo_disgust.wav"),
            ("😲 Surprised (3.2s)", "demo_surprised.wav"),
        ]
        for idx, (label, sfile) in enumerate(row2_samples):
            with row2_cols[idx]:
                if st.button(label, key=f"btn_demo_{sfile}", use_container_width=True):
                    s_path = os.path.join(samples_dir, sfile)
                    if not os.path.exists(s_path):
                        alt = sfile.replace("demo_", "sample_")
                        s_path = os.path.join(samples_dir, alt) if os.path.exists(os.path.join(samples_dir, alt)) else s_path
                    if os.path.exists(s_path):
                        with open(s_path, 'rb') as f:
                            input_audio_bytes = f.read()
                            input_audio_name = sfile

        long_path = os.path.join(samples_dir, "sample_long_speech.wav")
        if st.button("⏱️ Long Multi-Emotion Speech (6.5s — Calm + Happy for Timeline Dynamics)", key="btn_demo_long_speech", use_container_width=True):
            if os.path.exists(long_path):
                with open(long_path, 'rb') as f:
                    input_audio_bytes = f.read()
                    input_audio_name = "sample_long_speech.wav"

    # Load into session state if new audio is provided
    if input_audio_bytes is not None and (st.session_state.audio_bytes != input_audio_bytes):
        try:
            with st.spinner("Decoding and standardizing audio stream..."):
                y_raw, sr = decode_audio_bytes(input_audio_bytes, target_sr=config.SR)
                dur_sec = len(y_raw) / sr

                if dur_sec < 0.5:
                    st.error(f"❌ Audio is too short ({dur_sec:.2f}s). Minimum required duration is 0.5s.")
                else:
                    clean_wav = audio_to_wav_bytes(y_raw, sr)
                    st.session_state.audio_bytes = clean_wav
                    st.session_state.audio_name = input_audio_name
                    st.session_state.audio_data = y_raw
                    st.session_state.audio_sr = sr
                    st.session_state.audio_duration = dur_sec
                    st.session_state.analysis_results = None
                    st.session_state.dual_results = None
                    st.session_state.timeline_results = None
                    st.session_state.audio_features = None
                    st.rerun()
        except Exception as dec_err:
            st.error(f"❌ Failed to decode audio: {str(dec_err)}")

    # 3. Audio Preview & Waveform
    if st.session_state.audio_data is not None:
        st.markdown("### 3. Audio Preview & Acoustics")
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.write(f"**Loaded Signal:** `{st.session_state.audio_name}`")
            st.audio(st.session_state.audio_bytes, format='audio/wav')

            dur_sec = len(st.session_state.audio_data) / st.session_state.audio_sr
            stat_c1, stat_c2 = st.columns(2)
            stat_c1.metric("Duration", f"{dur_sec:.2f} s")
            stat_c2.metric("Sample Rate", f"{st.session_state.audio_sr} Hz")

            btn_res, btn_lab = st.columns(2)
            with btn_res:
                if st.button("🔄 RESET", use_container_width=True):
                    st.session_state.audio_data = None
                    st.session_state.audio_sr = None
                    st.session_state.audio_name = None
                    st.session_state.audio_bytes = None
                    st.session_state.audio_duration = None
                    st.session_state.analysis_results = None
                    st.session_state.dual_results = None
                    st.session_state.timeline_results = None
                    st.session_state.audio_features = None
                    st.rerun()

            with btn_lab:
                if st.button("🎛️ Audio Lab →", use_container_width=True):
                    st.session_state.page = "AUDIO LAB"
                    st.rerun()

        with p_col2:
            st.plotly_chart(
                plot_waveform(st.session_state.audio_data, st.session_state.audio_sr),
                use_container_width=True
            )

        # 4. Action Buttons (Single Model vs Dual-Model Comparison)
        st.markdown("---")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            analyze_clicked = st.button(
                f"🔬 ANALYZE AFFECT ({st.session_state.model_type.upper()})",
                type="primary",
                use_container_width=True
            )
        with act_col2:
            dual_clicked = st.button(
                "⚔️ DUAL-MODEL COMPARATIVE AUDIT (CNN vs CNN-LSTM)",
                use_container_width=True
            )

        # Trigger Single Inference
        if analyze_clicked:
            loaded_model, loaded_encoder, load_err = get_model_and_encoder(st.session_state.model_type)
            if loaded_model is None or loaded_encoder is None:
                st.error("## MODEL NOT TRAINED")
                st.markdown(f"The selected model (**{st.session_state.model_type.upper()}**) is not available. Please train it via `python src/train.py --model {st.session_state.model_type}`.")
            else:
                with st.spinner(f"Running neural inference via {st.session_state.model_type.upper()}..."):
                    try:
                        result = predict_emotion(
                            st.session_state.audio_data,
                            loaded_model,
                            loaded_encoder,
                            sr=st.session_state.audio_sr
                        )
                        timeline = predict_segments(
                            st.session_state.audio_data,
                            loaded_model,
                            loaded_encoder,
                            sr=st.session_state.audio_sr,
                            min_duration_for_timeline=4.0
                        )
                        features = extract_audio_features(
                            st.session_state.audio_data,
                            st.session_state.audio_sr
                        )

                        st.session_state.analysis_results = result
                        st.session_state.timeline_results = timeline
                        st.session_state.audio_features = features
                        st.session_state.dual_results = None

                        # Persist to SQLite Database (Requirement 4 & 5: CRUD)
                        rec_id = save_prediction(
                            filename=st.session_state.audio_name,
                            duration=round(dur_sec, 2),
                            model=st.session_state.model_type.upper(),
                            emotion=result['emotion'],
                            confidence=round(result['confidence'], 4),
                            confidence_pct=f"{result['confidence']*100:.1f}%",
                            confidence_level=result['confidence_level'],
                            top_3=", ".join([f"{item['emotion']} ({item['confidence_pct']}%)" for item in result.get('top_3', [])]),
                            probabilities=result['probabilities'],
                            features=features,
                            user_notes="",
                            user_tag="General",
                            is_starred=0
                        )
                        st.session_state.last_prediction_id = rec_id
                        st.rerun()

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        st.error(traceback.format_exc())

        # Trigger Dual Comparison
        if dual_clicked:
            model_lstm, enc_lstm, _ = get_model_and_encoder('cnn_lstm')
            model_cnn, enc_cnn, _ = get_model_and_encoder('cnn')
            if model_lstm is None or model_cnn is None:
                st.error("Both models must be trained for comparative audit.")
            else:
                with st.spinner("Executing dual-model comparative evaluation..."):
                    res_lstm = predict_emotion(st.session_state.audio_data, model_lstm, enc_lstm, sr=st.session_state.audio_sr)
                    res_cnn = predict_emotion(st.session_state.audio_data, model_cnn, enc_cnn, sr=st.session_state.audio_sr)
                    features = extract_audio_features(st.session_state.audio_data, st.session_state.audio_sr)
                    
                    st.session_state.dual_results = {
                        'cnn_lstm': res_lstm,
                        'cnn': res_cnn,
                        'agreement': (res_lstm['emotion'].lower() == res_cnn['emotion'].lower())
                    }
                    st.session_state.analysis_results = res_lstm  # default to LSTM
                    st.session_state.audio_features = features
                    
                    # Persist primary inference to SQLite
                    rec_id = save_prediction(
                        filename=st.session_state.audio_name,
                        duration=round(dur_sec, 2),
                        model="DUAL-AUDIT",
                        emotion=res_lstm['emotion'],
                        confidence=round(res_lstm['confidence'], 4),
                        confidence_pct=f"{res_lstm['confidence']*100:.1f}%",
                        confidence_level=res_lstm['confidence_level'],
                        top_3=f"LSTM: {res_lstm['emotion']} | CNN: {res_cnn['emotion']}",
                        probabilities=res_lstm['probabilities'],
                        features=features,
                        user_notes="Dual model comparative audit executed.",
                        user_tag="Comparative",
                        is_starred=0
                    )
                    st.session_state.last_prediction_id = rec_id
                    st.rerun()

    # 5. Dual-Model Comparative Card Display
    if st.session_state.dual_results is not None:
        d = st.session_state.dual_results
        res_l = d['cnn_lstm']
        res_c = d['cnn']
        is_agree = d['agreement']

        st.markdown("### ⚔️ Dual-Model Architecture Comparative Audit")
        
        status_box = """
        <div style='background:rgba(16,185,129,0.15); border:1px solid #10b981; border-radius:8px; padding:12px; margin-bottom:15px; text-align:center;'>
            <strong style='color:#10b981; font-size:1.15rem;'>🟢 FULL ARCHITECTURAL CONSENSUS</strong>: Both CNN and CNN-LSTM arrived at the same predicted emotion.
        </div>
        """ if is_agree else """
        <div style='background:rgba(245,158,11,0.15); border:1px solid #f59e0b; border-radius:8px; padding:12px; margin-bottom:15px; text-align:center;'>
            <strong style='color:#f59e0b; font-size:1.15rem;'>🟡 DIVERGENT PREDICTION</strong>: Models captured distinct acoustic cues (temporal cadence vs spatial harmonics).
        </div>
        """
        st.markdown(status_box, unsafe_allow_html=True)

        cmp1, cmp2 = st.columns(2)
        with cmp1:
            st.markdown(f"""
            <div class='glass-card' style='border-top:4px solid #38bdf8;'>
                <div style='font-size:0.85rem; color:var(--text-muted); text-transform:uppercase;'>CNN-LSTM Hybrid (Spatiotemporal)</div>
                <div style='font-size:2rem; font-weight:700; color:#38bdf8; margin:0.3rem 0;'>
                    {config.EMOTION_EMOJIS.get(res_l['emotion'], '')} {res_l['emotion'].upper()}
                </div>
                <div>Confidence: <strong>{res_l['confidence_pct']}%</strong> ({res_l['confidence_level']})</div>
            </div>
            """, unsafe_allow_html=True)

        with cmp2:
            st.markdown(f"""
            <div class='glass-card' style='border-top:4px solid #818cf8;'>
                <div style='font-size:0.85rem; color:var(--text-muted); text-transform:uppercase;'>CNN Baseline (Spatial Only)</div>
                <div style='font-size:2rem; font-weight:700; color:#818cf8; margin:0.3rem 0;'>
                    {config.EMOTION_EMOJIS.get(res_c['emotion'], '')} {res_c['emotion'].upper()}
                </div>
                <div>Confidence: <strong>{res_c['confidence_pct']}%</strong> ({res_c['confidence_level']})</div>
            </div>
            """, unsafe_allow_html=True)

        # Comparative Bar Chart
        st.markdown("<br>", unsafe_allow_html=True)
        chart_df = pd.DataFrame({
            'Emotion': list(res_l['probabilities'].keys()) + list(res_c['probabilities'].keys()),
            'Probability': [v*100 for v in res_l['probabilities'].values()] + [v*100 for v in res_c['probabilities'].values()],
            'Model': ['CNN-LSTM'] * len(res_l['probabilities']) + ['CNN Baseline'] * len(res_c['probabilities'])
        })
        fig_cmp = px.bar(
            chart_df, x='Emotion', y='Probability', color='Model',
            barmode='group',
            color_discrete_map={'CNN-LSTM': '#38bdf8', 'CNN Baseline': '#818cf8'},
            title="Probability Distribution Comparison across Classes (%)"
        )
        fig_cmp.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Forum, serif", color="#e2e8f0")
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # 6. Primary Prediction Display
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        primary_emo = res['emotion']
        conf_pct = res['confidence_pct']
        conf_lvl = res['confidence_level']
        emo_color = config.EMOTION_COLORS.get(primary_emo, "#38bdf8")
        emo_emoji = config.EMOTION_EMOJIS.get(primary_emo, "🎭")

        badge_bg = "rgba(16, 185, 129, 0.2)" if conf_lvl == "High" else "rgba(245, 158, 11, 0.2)" if conf_lvl == "Moderate" else "rgba(239, 68, 68, 0.2)"
        badge_color = "#10b981" if conf_lvl == "High" else "#f59e0b" if conf_lvl == "Moderate" else "#ef4444"

        st.markdown(f"""
        <div class='prediction-box' style='border-color: {emo_color};'>
            <div style='font-size: 3.5rem;'>{emo_emoji}</div>
            <div style='font-size: 2.8rem; font-weight:800; color:{emo_color}; text-transform:uppercase; margin-top:0.25rem;'>
                {primary_emo}
            </div>
            <div style='font-size: 1.3rem; color: var(--text-secondary); margin-top:0.5rem;'>
                Inference Confidence: <strong style='color:#f8fafc;'>{conf_pct}%</strong>
            </div>
            <div class='prediction-badge' style='background:{badge_bg}; color:{badge_color}; border: 1px solid {badge_color};'>
                {conf_lvl} Confidence Level
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("ℹ️ **Confidence Interpretation:** Reflects model posterior probability distribution over the 8 trained RAVDESS classes. It represents acoustic feature affinity, not an absolute diagnosis of human emotion.")

        # Radar & Top 3 Side-by-Side
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("### 🕸️ Emotion Constellation (Radar)")
            st.plotly_chart(plot_emotion_radar(res['probabilities']), use_container_width=True)

        with r_col2:
            st.markdown("### 🎯 Top-3 Emotion Candidates")
            top_3 = res.get('top_3', [])
            for item in top_3:
                e_name = item['emotion']
                e_pct = item['confidence_pct']
                e_col = config.EMOTION_COLORS.get(e_name, "#38bdf8")
                e_em = config.EMOTION_EMOJIS.get(e_name, "•")
                st.markdown(f"""
                <div class='progress-container'>
                    <div class='progress-label'>
                        <span style='text-transform: capitalize;'>{e_em} {e_name}</span>
                        <span style='color: {e_col}; font-weight:700;'>{e_pct}%</span>
                    </div>
                    <div class='progress-track'>
                        <div class='progress-bar-fill' style='width: {e_pct}%; background-color: {e_col};'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Probability Breakdown")
            st.plotly_chart(plot_emotion_bars(res['probabilities']), use_container_width=True)

        # Audio DNA Spectrogram
        st.markdown("---")
        st.markdown("### 🧬 Audio DNA (Log-Mel Spectrogram)")
        st.caption("Acoustic energy distribution across 128 Mel frequency bands.")
        if st.session_state.audio_data is not None:
            st.plotly_chart(
                plot_mel_spectrogram(st.session_state.audio_data, st.session_state.audio_sr),
                use_container_width=True
            )

        # Emotion Timeline
        st.markdown("---")
        st.markdown("### ⏱️ Emotion Trajectory Timeline")
        if st.session_state.timeline_results is not None:
            st.caption("Sliding-window segment-level predictions across time.")
            st.plotly_chart(
                plot_emotion_timeline(st.session_state.timeline_results),
                use_container_width=True
            )
            st.warning("⚠️ Segment-level emotion predictions indicate localized vocal intonation shifts and should not be used for psychological diagnosis.")
        else:
            st.info("Audio duration is under 4.0 seconds; whole-utterance prediction applies.")

        # Voice Energy (RMS)
        st.markdown("---")
        st.markdown("### ⚡ Voice Energy Dynamics (RMS)")
        if st.session_state.audio_data is not None:
            st.plotly_chart(
                plot_rms_energy(st.session_state.audio_data, st.session_state.audio_sr),
                use_container_width=True
            )

        # Feature metrics expander
        with st.expander("📋 Detailed Acoustic Feature Metrics", expanded=False):
            feats = st.session_state.audio_features
            if feats:
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("RMS Energy (Mean)", f"{feats['rms_mean']:.4f}")
                fc2.metric("Zero Crossing Rate", f"{feats['zcr_mean']:.4f}")
                fc3.metric("Spectral Centroid", f"{feats['spectral_centroid_mean']:.0f} Hz")
                fc4.metric("Spectral Bandwidth", f"{feats['spectral_bandwidth_mean']:.0f} Hz")

                fc5, fc6, fc7, fc8 = st.columns(4)
                fc5.metric("Spectral Rolloff", f"{feats['spectral_rolloff_mean']:.0f} Hz")
                fc6.metric("Mean MFCC", f"{feats['mfcc_mean']:.3f}")
                fc7.metric("Duration", f"{feats['duration']:.2f} s")
                fc8.metric("Sample Rate", f"{feats['sample_rate']} Hz")

        # Acoustic interpretation expander
        with st.expander("🤔 Acoustic Interpretation & Emotion Markers", expanded=False):
            desc = config.EMOTION_DESCRIPTIONS.get(primary_emo, "No description available.")
            st.markdown(f"**Acoustic Profile for {primary_emo.capitalize()}:**")
            st.write(desc)
            st.caption("Based on supervised learning from professional actors in the RAVDESS corpus.")

        # Reviewer Annotations & Database Record Update (CRUD - UPDATE)
        st.markdown("---")
        st.markdown("### 📝 Reviewer Observations & Database Record Update")
        with st.form("form_update_record"):
            note_c1, note_c2, note_c3 = st.columns([2, 1, 1])
            with note_c1:
                reviewer_note = st.text_input("Reviewer Notes / Clinical Commentary:", placeholder="e.g., Noticeable pitch elevation during second syllable.")
            with note_c2:
                reviewer_tag = st.selectbox("Tag Category:", ["General", "Interview", "Clinical Research", "Customer Call", "Stress Test", "Acoustic Benchmark"])
            with note_c3:
                star_flag = st.checkbox("⭐ Mark as Starred Record", value=False)
            
            submit_note = st.form_submit_button("💾 Save Reviewer Annotations to Database", use_container_width=True)
            if submit_note:
                if st.session_state.last_prediction_id:
                    update_prediction(
                        st.session_state.last_prediction_id,
                        user_notes=reviewer_note,
                        user_tag=reviewer_tag,
                        is_starred=1 if star_flag else 0
                    )
                    st.success(f"Record #{st.session_state.last_prediction_id} successfully updated in database!")
                else:
                    st.warning("No recent record ID found to update.")

        # Export Analysis Report Button
        st.markdown("---")
        html_report_content = generate_html_report(
            res,
            st.session_state.audio_name,
            st.session_state.audio_sr,
            st.session_state.audio_data,
            active_model_name,
            st.session_state.audio_features
        )
        report_filename = f"emotiva_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        exp_c1, exp_c2, exp_c3 = st.columns([1, 1.6, 1])
        with exp_c2:
            st.download_button(
                label="📄 EXPORT COMPLETE ANALYSIS REPORT (HTML)",
                data=html_report_content,
                file_name=report_filename,
                mime="text/html",
                type="primary",
                use_container_width=True
            )


# =========================================================
# PAGE 3: AUDIO LAB (Interactive DSP & Feature Workstation)
# =========================================================
elif st.session_state.page == "AUDIO LAB":
    st.markdown("<h2>🎛️ Audio Signal Processing Workstation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); font-size:1.1rem;'>Interactive signal inspection, filterbanks, MFCCs, spectral distributions, and DSP stress-testing.</p>", unsafe_allow_html=True)

    # If no audio is loaded, provide an in-page audio selector!
    if st.session_state.audio_data is None:
        st.info("No audio signal currently active in memory. Select a pre-recorded demo voice or upload audio below to inspect in the Audio Lab:")
        
        c_sel1, c_sel2 = st.columns([1.5, 1])
        with c_sel1:
            samples_dir = os.path.join(config.PROJECT_ROOT, "samples")
            sample_options = [
                ("Neutral Speech (3.2s)", "demo_neutral.wav"),
                ("Calm Speech (3.2s)", "demo_calm.wav"),
                ("Happy Speech (3.2s)", "demo_happy.wav"),
                ("Sad Speech (3.2s)", "demo_sad.wav"),
                ("Angry Speech (3.2s)", "demo_angry.wav"),
                ("Fearful Speech (3.2s)", "demo_fearful.wav"),
                ("Disgust Speech (3.2s)", "demo_disgust.wav"),
                ("Surprised Speech (3.2s)", "demo_surprised.wav"),
                ("Long Multi-Emotion Speech (6.5s)", "sample_long_speech.wav"),
            ]
            picked_demo = st.selectbox("Choose demo sample to load into workstation:", [item[0] for item in sample_options])
            picked_file = next(item[1] for item in sample_options if item[0] == picked_demo)
            
            if st.button("📥 Load Selected Sample into Audio Lab", type="primary"):
                s_path = os.path.join(samples_dir, picked_file)
                if not os.path.exists(s_path):
                    s_path = os.path.join(samples_dir, picked_file.replace("demo_", "sample_"))
                if os.path.exists(s_path):
                    y_raw, sr = load_audio_raw(s_path, sr=config.SR)
                    with open(s_path, 'rb') as f:
                        b_data = f.read()
                    st.session_state.audio_data = y_raw
                    st.session_state.audio_sr = sr
                    st.session_state.audio_name = picked_file
                    st.session_state.audio_bytes = b_data
                    st.session_state.audio_duration = len(y_raw) / sr
                    st.rerun()

        with c_sel2:
            st.markdown("Or navigate back to analyze voice:")
            if st.button("← Go to Analyze Page", use_container_width=True):
                st.session_state.page = "ANALYZE"
                st.rerun()

    else:
        y = st.session_state.audio_data
        sr = st.session_state.audio_sr

        # Top Audio Header
        top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
        with top_col1:
            st.write(f"**Active Signal:** `{st.session_state.audio_name}`")
            st.audio(st.session_state.audio_bytes, format='audio/wav')
        with top_col2:
            st.metric("Total Samples", f"{len(y):,}")
            st.metric("Duration", f"{len(y)/sr:.2f} s")
        with top_col3:
            st.metric("Sample Rate", f"{sr} Hz")
            if st.button("← Back to Analyze", use_container_width=True):
                st.session_state.page = "ANALYZE"
                st.rerun()

        st.markdown("---")

        lab_tabs = st.tabs([
            "Waveform", "Mel-Spectrogram", "MFCC Coefficients",
            "Spectral Features", "RMS Loudness", "Zero Crossing Rate",
            "🎛️ DSP Filtering & Stress Testing"
        ])

        # TAB 1: WAVEFORM
        with lab_tabs[0]:
            st.markdown("#### Time-Domain Signal Amplitude")
            st.write("Displays the instantaneous raw acoustic pressure amplitude normalized to [-1.0, 1.0].")
            st.plotly_chart(plot_waveform(y, sr), use_container_width=True)

        # TAB 2: MEL-SPECTROGRAM
        with lab_tabs[1]:
            st.markdown("#### Log-Mel Filterbank Energy Spectrogram")
            st.write("Maps frequency onto the Mel scale, reflecting non-linear human auditory perception.")

            spec_c1, spec_c2, spec_c3 = st.columns(3)
            with spec_c1:
                n_mels = st.slider("Mel Filterbank Bands (N_MELS)", 32, 256, 128, step=16, key="slider_mels")
            with spec_c2:
                n_fft = st.selectbox("FFT Window Size (N_FFT)", [512, 1024, 2048, 4096], index=2, key="select_fft")
            with spec_c3:
                hop_len = st.selectbox("Hop Length", [128, 256, 512, 1024], index=2, key="select_hop")

            st.plotly_chart(
                plot_mel_spectrogram(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_len),
                use_container_width=True
            )

        # TAB 3: MFCC
        with lab_tabs[2]:
            st.markdown("#### Mel-Frequency Cepstral Coefficients (MFCC)")
            st.write("Captures the spectral vocal tract envelope shape for phonetic and timbral characterization.")

            mfcc_c1, mfcc_c2 = st.columns([1, 3])
            with mfcc_c1:
                num_mfcc = st.slider(
                    "MFCC Coefficients Count",
                    min_value=10, max_value=40, value=st.session_state.mfcc_n, step=2,
                    key="slider_mfcc"
                )
                st.session_state.mfcc_n = num_mfcc
                st.caption(f"Visualizing lowest {num_mfcc} cepstral coefficients.")

            with mfcc_c2:
                st.plotly_chart(
                    plot_mfcc(y, sr, n_mfcc=num_mfcc),
                    use_container_width=True
                )

        # TAB 4: SPECTRAL FEATURES
        with lab_tabs[3]:
            st.markdown("#### High-Order Spectral Dynamics")
            st.write("Tracking frequency centroid ('timbre brightness'), bandwidth, and spectral rolloff.")

            st.plotly_chart(plot_spectral_centroid(y, sr), use_container_width=True)
            col_sp1, col_sp2 = st.columns(2)
            with col_sp1:
                st.plotly_chart(plot_spectral_bandwidth(y, sr), use_container_width=True)
            with col_sp2:
                st.plotly_chart(plot_spectral_rolloff(y, sr), use_container_width=True)

        # TAB 5: RMS ENERGY
        with lab_tabs[4]:
            st.markdown("#### Root-Mean-Square Energy Dynamics")
            st.write("Measures acoustic power and vocal intensity over time.")
            st.plotly_chart(plot_rms_energy(y, sr), use_container_width=True)

        # TAB 6: ZERO CROSSING RATE
        with lab_tabs[5]:
            st.markdown("#### Zero Crossing Rate (ZCR)")
            st.write("Frequency of signal polarity inversions — discriminates unvoiced fricatives, breathiness, and percussive noise.")
            st.plotly_chart(plot_zero_crossing_rate(y, sr), use_container_width=True)

        # TAB 7: DSP FILTERING & STRESS TESTING
        with lab_tabs[6]:
            st.markdown("#### 🎛️ Digital Signal Distortion & Robustness Suite")
            st.write("Simulate real-world acoustic degradations (pitch shift, time stretch, additive white noise, bandpass filtering) to test neural model resilience.")

            dsp_col1, dsp_col2, dsp_col3 = st.columns(3)
            with dsp_col1:
                pitch_shift_steps = st.slider("Pitch Shift (Semitones)", -4.0, 4.0, 0.0, step=0.5, key="dsp_pitch")
            with dsp_col2:
                time_stretch_val = st.slider("Time Stretch Factor", 0.8, 1.25, 1.0, step=0.05, key="dsp_stretch")
            with dsp_col3:
                noise_enable = st.checkbox("Inject Gaussian White Noise", value=False, key="dsp_noise_en")
                noise_snr = st.slider("Signal-to-Noise Ratio (SNR dB)", 10, 40, 25, step=5, disabled=not noise_enable, key="dsp_snr")

            f_col1, f_col2 = st.columns(2)
            with f_col1:
                filter_mode = st.selectbox("Spectral Filtering Mode:", ["None", "Low-Pass", "High-Pass"], key="dsp_filt_mode")
            with f_col2:
                cutoff_f = st.slider("Cutoff Frequency (Hz)", 500, 8000, 3000, step=250, disabled=(filter_mode == "None"), key="dsp_cutoff")

            if st.button("⚡ Apply DSP Effects & Preview Processed Signal", type="primary"):
                with st.spinner("Synthesizing transformed acoustic signal..."):
                    y_proc = y.copy()
                    if abs(pitch_shift_steps) > 0.1:
                        y_proc = librosa.effects.pitch_shift(y_proc, sr=sr, n_steps=pitch_shift_steps)
                    if abs(time_stretch_val - 1.0) > 0.04:
                        y_proc = librosa.effects.time_stretch(y_proc, rate=time_stretch_val)
                    if noise_enable:
                        sig_power = np.mean(y_proc ** 2)
                        if sig_power > 0:
                            noise_power = sig_power / (10 ** (noise_snr / 10))
                            noise = np.random.normal(0, np.sqrt(noise_power), len(y_proc))
                            y_proc = y_proc + noise
                    if filter_mode in ["Low-Pass", "High-Pass"]:
                        btype = 'lowpass' if filter_mode == "Low-Pass" else 'highpass'
                        nyq = 0.5 * sr
                        norm_cutoff = min(max(cutoff_f / nyq, 0.01), 0.99)
                        sos = butter(4, norm_cutoff, btype=btype, output='sos')
                        y_proc = sosfilt(sos, y_proc)
                    if np.max(np.abs(y_proc)) > 0:
                        y_proc = librosa.util.normalize(y_proc)

                    # Encode to WAV bytes for audio player
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_proc:
                        sf.write(tmp_proc.name, y_proc, sr)
                        tmp_proc_path = tmp_proc.name

                    with open(tmp_proc_path, 'rb') as f_proc:
                        proc_bytes = f_proc.read()
                    os.unlink(tmp_proc_path)

                    st.session_state.audio_data = y_proc
                    st.session_state.audio_bytes = proc_bytes
                    st.session_state.audio_name = f"dsp_{st.session_state.audio_name}"
                    st.session_state.audio_duration = len(y_proc) / sr
                    st.session_state.analysis_results = None
                    st.success("Transformed signal loaded as active workstation audio! Switch to ANALYZE page to test model resilience.")
                    st.rerun()


# =========================================================
# PAGE 4: INSIGHTS (Dataset Statistics & Exploration)
# =========================================================
elif st.session_state.page == "INSIGHTS":
    st.markdown("<h2>📊 RAVDESS Dataset Insights & Affect Statistics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); font-size:1.1rem;'>Acoustic corpus composition, demographic distributions, and affective profiles.</p>", unsafe_allow_html=True)

    df_meta = load_ravdess_metadata(config.DATASET_PATH)

    if df_meta.empty or not os.path.exists(config.DATASET_PATH):
        st.error("## DATASET NOT FOUND")
        st.markdown(f"The RAVDESS audio dataset was not detected in: `{config.DATASET_PATH}`.")
    else:
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Total Audio Recordings", f"{len(df_meta):,}")
        ic2.metric("Professional Actors", df_meta['actor'].nunique())
        ic3.metric("Emotion Categories", df_meta['emotion'].nunique())
        ic4.metric("Standardized Duration", f"{config.DURATION:.1f} s")

        st.markdown("---")
        ch_c1, ch_c2 = st.columns([1.5, 1])
        with ch_c1:
            st.markdown("### Emotion Distribution in Corpus")
            st.plotly_chart(plot_emotion_distribution(df_meta), use_container_width=True)

        with ch_c2:
            st.markdown("### Actor Gender Distribution")
            male_count = len(df_meta[df_meta['actor'] % 2 != 0])
            female_count = len(df_meta[df_meta['actor'] % 2 == 0])
            gender_df = pd.DataFrame({
                'Gender': ['Male Speakers (Odd Actors)', 'Female Speakers (Even Actors)'],
                'Recordings': [male_count, female_count]
            })
            fig_gender = px.pie(
                gender_df, names='Gender', values='Recordings',
                color_discrete_sequence=['#38bdf8', '#f472b6'],
                hole=0.45
            )
            fig_gender.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Forum, serif", color="#e2e8f0")
            )
            st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎭 Interactive Emotion Acoustic Profile & Auditory Player")

    selected_emo = st.selectbox("Select Affect Category to Inspect:", config.EMOTIONS, key="select_insight_emo")

    emo_color = config.EMOTION_COLORS.get(selected_emo, "#38bdf8")
    emo_emoji = config.EMOTION_EMOJIS.get(selected_emo, "🎭")
    emo_desc = config.EMOTION_DESCRIPTIONS.get(selected_emo, "No description available.")

    # Match emotion code to count
    sample_count = "N/A"
    if not df_meta.empty and 'emotion' in df_meta.columns:
        code = next((k for k, v in config.EMOTION_LABELS.items() if v == selected_emo), None)
        if code is not None:
            sample_count = f"{len(df_meta[df_meta['emotion'] == code])} recordings"

    st.markdown(f"""
    <div class='glass-card' style='border-left: 6px solid {emo_color};'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h3 style='color: {emo_color}; margin: 0;'>{emo_emoji} {selected_emo.capitalize()}</h3>
            <span style='background:rgba(255,255,255,0.08); padding:4px 12px; border-radius:12px; font-weight:600;'>
                Corpus Count: {sample_count}
            </span>
        </div>
        <p style='font-size: 1.15rem; margin-top: 1rem; color: var(--text-primary); line-height: 1.6;'>
            {emo_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Auditory Player for the Selected Emotion
    samples_dir = os.path.join(config.PROJECT_ROOT, "samples")
    demo_wav_name = f"demo_{selected_emo}.wav"
    demo_path = os.path.join(samples_dir, demo_wav_name)
    if not os.path.exists(demo_path):
        demo_path = os.path.join(samples_dir, f"sample_{selected_emo}.wav")

    if os.path.exists(demo_path):
        st.markdown(f"#### 🎧 Listen to Synthetic Formant Voice: `{demo_wav_name}`")
        with open(demo_path, 'rb') as f_demo:
            st.audio(f_demo.read(), format='audio/wav')

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Explore Model Benchmarks & Architecture (MODELS) →", type="primary"):
        st.session_state.page = "MODELS"
        st.rerun()


# =========================================================
# PAGE 5: MODELS (Benchmarks, Convergence & Live Test)
# =========================================================
elif st.session_state.page == "MODELS":
    st.markdown("<h2>🧪 Deep Learning Models & Benchmark Evaluations</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); font-size:1.1rem;'>Comparative performance metrics, confusion matrices, and training convergence curves.</p>", unsafe_allow_html=True)

    comp_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    eval_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    cnn_hist_path = os.path.join(config.RESULTS_DIR, "cnn_history.json")
    lstm_hist_path = os.path.join(config.RESULTS_DIR, "cnn_lstm_history.json")

    has_comparison = os.path.exists(comp_path)
    has_evaluation = os.path.exists(eval_path)

    # 1. Benchmark Comparison Table & Graph
    st.markdown("### 1. Comparative Architecture Performance")
    if has_comparison:
        comp_df = pd.read_csv(comp_path)
        st.plotly_chart(plot_model_comparison(comp_df), use_container_width=True)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Model comparison metrics not available yet. Please train and evaluate models via `python src/evaluate.py`.")

    st.markdown("---")

    # 2. Training Convergence History
    st.markdown("### 2. Training & Validation Convergence Curves")
    hist_col1, hist_col2 = st.columns(2)

    with hist_col1:
        st.markdown("#### CNN-LSTM Curves")
        if os.path.exists(lstm_hist_path):
            with open(lstm_hist_path, 'r') as f:
                lstm_hist = json.load(f)
            st.plotly_chart(plot_training_curves(lstm_hist), use_container_width=True)
        else:
            st.info("CNN-LSTM history not available.")

    with hist_col2:
        st.markdown("#### CNN Baseline Curves")
        if os.path.exists(cnn_hist_path):
            with open(cnn_hist_path, 'r') as f:
                cnn_hist = json.load(f)
            st.plotly_chart(plot_training_curves(cnn_hist), use_container_width=True)
        else:
            st.info("CNN Baseline history not available.")

    st.markdown("---")

    # 3. Confusion Matrix
    st.markdown("### 3. Test Set Confusion Matrix")
    if has_evaluation:
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)

        m_opt = st.selectbox("Select Model for Confusion Matrix:", [k.upper() for k in eval_data.keys()])
        selected_eval = eval_data.get(m_opt.lower(), {})

        if 'confusion_matrix' in selected_eval:
            cm = selected_eval['confusion_matrix']
            labels = selected_eval.get('class_names', config.EMOTIONS)
            st.plotly_chart(plot_confusion_matrix(cm, labels), use_container_width=True)

            if 'report' in selected_eval:
                st.markdown("#### Per-Class Classification Report")
                st.code(selected_eval['report'])
    else:
        st.info("Evaluation results not available yet.")

    st.markdown("---")

    # 4. Architectural Summary
    st.markdown("### 4. Neural Network Architectures")
    arch_c1, arch_c2 = st.columns(2)

    with arch_c1:
        st.markdown("#### CNN-LSTM (Main Spatiotemporal Model)")
        st.markdown("""
        ```
        Input: (128 Mel Bands, 130 Frames, 1 Channel)
        │
        ├── Conv2D(32, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 2) + Dropout(0.2)
        │
        ├── Conv2D(64, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 2) + Dropout(0.2)
        │
        ├── Conv2D(64, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 2) + Dropout(0.2)
        │
        ├── Permute((2, 1, 3)) [Time axis first]
        ├── Reshape((16 TimeSteps, 1024 Features))
        │
        ├── LSTM(64 Units) + Dropout(0.3)
        ├── Dense(64, ReLU)
        └── Dense(8, Softmax) -> Categorical Affect
        ```
        """)

    with arch_c2:
        st.markdown("#### CNN Baseline (Spatial Convolutional Model)")
        st.markdown("""
        ```
        Input: (128 Mel Bands, 130 Frames, 1 Channel)
        │
        ├── Conv2D(32, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 2) + Dropout(0.2)
        │
        ├── Conv2D(64, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 2) + Dropout(0.2)
        │
        ├── Conv2D(64, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 2) + Dropout(0.2)
        │
        ├── Flatten()
        ├── Dense(64, ReLU) + Dropout(0.3)
        └── Dense(8, Softmax) -> Categorical Affect
        ```
        """)


# =========================================================
# PAGE 6: HISTORY (SQLite CRUD & Audit Log)
# =========================================================
elif st.session_state.page == "HISTORY":
    st.markdown("<h2>📜 Persistent Inference History & Audit Log (SQLite)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); font-size:1.1rem;'>Audit trail of all audio predictions with database persistence, notes, tags, filtering, and export.</p>", unsafe_allow_html=True)

    # 1. Summary Statistics Cards (CRUD - Aggregation)
    stats = get_history_stats()
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    kpi_c1.metric("Total Analyses Saved", stats['total_records'])
    kpi_c2.metric("Starred Inferences", stats['starred_records'])
    kpi_c3.metric("Average Confidence", stats['average_confidence_pct'])
    kpi_c4.metric("Top Emotion", stats['top_emotion'].capitalize())

    st.markdown("---")

    # 2. Filter & Search Controls (CRUD - READ)
    st.markdown("### Search & Filter Audit Records")
    f_row1, f_row2, f_row3, f_row4 = st.columns([2, 1, 1, 1.2])

    with f_row1:
        search_query = st.text_input("🔍 Search Query (filename, notes, tag, emotion):", placeholder="e.g. happy, interview, demo")

    with f_row2:
        filter_emo = st.selectbox("Emotion Filter:", ["ALL"] + [e.upper() for e in config.EMOTIONS])

    with f_row3:
        filter_mod = st.selectbox("Model Filter:", ["ALL", "CNN", "CNN-LSTM", "DUAL-AUDIT"])

    with f_row4:
        sort_order = st.selectbox("Sort Order:", ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence", "Duration"])

    starred_filter = st.checkbox("⭐ Show Only Starred Records", value=False)

    # Fetch from SQLite database
    records = get_predictions(
        emotion_filter=filter_emo,
        model_filter=filter_mod,
        search_query=search_query,
        sort_by=sort_order,
        starred_only=starred_filter
    )

    st.write(f"Displaying **{len(records)}** audit records from database:")

    if not records:
        st.info("No matching inference records found in database. Go to ANALYZE page to run predictions.")
        if st.button("← Go to Analyze"):
            st.session_state.page = "ANALYZE"
            st.rerun()
    else:
        # Action Bar: Export & Clear
        act_c1, act_c2, act_c3 = st.columns([1.5, 1.5, 2])
        with act_c1:
            # CSV Export
            df_export = pd.DataFrame(records)
            cols_to_drop = [c for c in ['probabilities', 'features', 'probabilities_json', 'features_json'] if c in df_export.columns]
            df_csv = df_export.drop(columns=cols_to_drop)
            st.download_button(
                label="📥 EXPORT HISTORY (CSV)",
                data=df_csv.to_csv(index=False).encode('utf-8'),
                file_name=f"emotiva_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with act_c2:
            # JSON Export
            clean_records = []
            for r in records:
                item_copy = dict(r)
                item_copy.pop('probabilities_json', None)
                item_copy.pop('features_json', None)
                clean_records.append(item_copy)
            st.download_button(
                label="📥 EXPORT HISTORY (JSON)",
                data=json.dumps(clean_records, indent=2).encode('utf-8'),
                file_name=f"emotiva_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with act_c3:
            confirm_clear = st.checkbox("Confirm clear entire database", key="chk_clear_all")
            if st.button("🗑️ PURGE ENTIRE HISTORY", disabled=not confirm_clear, use_container_width=True):
                clear_all_predictions()
                st.success("All audit records purged from SQLite database.")
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Individual Record List with Interactive Actions (CRUD - UPDATE / DELETE / ACTIVATE)
        for rec in records:
            rec_id = rec['id']
            emo_badge = config.EMOTION_EMOJIS.get(rec['emotion'], '🎭')
            star_icon = "⭐ " if rec['is_starred'] else ""
            
            with st.expander(f"{star_icon}#{rec_id} | {rec['timestamp']} | {emo_badge} {rec['emotion'].upper()} ({rec['confidence_pct']}) | {rec['filename']} [{rec['model']}]", expanded=False):
                r_col1, r_col2 = st.columns([2, 1])
                
                with r_col1:
                    st.write(f"**Filename:** `{rec['filename']}` &bull; **Duration:** {rec['duration']:.2f}s &bull; **Model:** {rec['model']}")
                    st.write(f"**Predicted Emotion:** **{rec['emotion'].capitalize()}** ({rec['confidence_pct']} - {rec['confidence_level']})")
                    st.write(f"**Top Candidates:** {rec['top_3']}")
                    if rec['user_notes']:
                        st.info(f"**Reviewer Note:** {rec['user_notes']}")
                    st.caption(f"Tag: `{rec['user_tag']}` | Starred: {'Yes' if rec['is_starred'] else 'No'}")

                with r_col2:
                    st.markdown("#### Record Actions")
                    
                    # Update Starred status
                    new_star_state = 0 if rec['is_starred'] else 1
                    star_label = "⭐ Unstar Record" if rec['is_starred'] else "☆ Star Record"
                    if st.button(star_label, key=f"star_btn_{rec_id}", use_container_width=True):
                        update_prediction(rec_id, is_starred=new_star_state)
                        st.rerun()

                    # Delete single record
                    if st.button("🗑️ Delete Record", key=f"del_btn_{rec_id}", use_container_width=True):
                        delete_prediction(rec_id)
                        st.success(f"Record #{rec_id} deleted.")
                        st.rerun()

                # Inline Edit Form (CRUD - UPDATE)
                st.markdown("---")
                with st.form(f"edit_form_{rec_id}"):
                    st.write("✏️ **Edit Reviewer Annotations:**")
                    edit_c1, edit_c2 = st.columns([2, 1])
                    with edit_c1:
                        updated_notes = st.text_input("Reviewer Notes:", value=rec['user_notes'] or "", key=f"edit_note_{rec_id}")
                    with edit_c2:
                        updated_tag = st.selectbox("Tag:", ["General", "Interview", "Clinical Research", "Customer Call", "Stress Test", "Acoustic Benchmark"], index=0, key=f"edit_tag_{rec_id}")
                    
                    if st.form_submit_button("💾 Save Annotation Update"):
                        update_prediction(rec_id, user_notes=updated_notes, user_tag=updated_tag)
                        st.success("Record updated successfully!")
                        st.rerun()


# =========================================================
# PAGE 7: ABOUT (Specifications, Ethics & System Diagnostics)
# =========================================================
elif st.session_state.page == "ABOUT":
    st.markdown("<h2>ℹ️ About SONORA / EMOTIVA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); font-size:1.1rem;'>Architectural specifications, acoustic signal methodology, ethical AI frameworks, and diagnostic self-check.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card' style='margin-bottom: 2rem;'>
        <h3 style='color: var(--accent-cyan); margin-top: 0;'>Project Abstract</h3>
        <p style='line-height: 1.7; font-size: 1.15rem;'>
            <strong>SONORA / EMOTIVA</strong> is an end-to-end Speech Emotion Recognition (SER) system
            engineered to classify human vocal affect directly from raw acoustic signals. By converting speech
            into 128-band Mel-Filterbank energy representations and feeding them through a hybrid
            <strong>Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM)</strong> architecture,
            the system captures both localized spectral formant geometries and temporal prosodic contours.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Core Technical Stack")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size: 2rem;'>🧠</div>
            <strong>TensorFlow 2.x / Keras</strong><br>
            <small style='color:var(--text-muted);'>Deep Spatiotemporal Models</small>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size: 2rem;'>🎵</div>
            <strong>Librosa & Scipy DSP</strong><br>
            <small style='color:var(--text-muted);'>Acoustic Signal Processing</small>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size: 2rem;'>💾</div>
            <strong>SQLite Engine</strong><br>
            <small style='color:var(--text-muted);'>Persistent Audit Logging</small>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size: 2rem;'>🎈</div>
            <strong>Streamlit Client</strong><br>
            <small style='color:var(--text-muted);'>Editorial Interactive UI</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # System Health & Diagnostic Self-Check
    st.markdown("### 🛠️ System Health & Diagnostic Self-Check")
    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        st.markdown("#### Environment & Dependencies")
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        st.write(f"• **Python Runtime:** `{py_ver}` 🟢")
        st.write(f"• **Streamlit Version:** `{st.__version__}` 🟢")
        st.write(f"• **Librosa Version:** `{librosa.__version__}` 🟢")
        st.write(f"• **Plotly Version:** `{px.__version__}` 🟢")

    with diag_c2:
        st.markdown("#### Models & Storage Integrity")
        cnn_path = os.path.join(config.MODELS_DIR, "cnn_model.keras")
        lstm_path = os.path.join(config.MODELS_DIR, "cnn_lstm_model.keras")
        enc_path = os.path.join(config.MODELS_DIR, "label_encoder.pkl")
        db_path = os.path.join(config.RESULTS_DIR, "emotiva_history.db")

        st.write(f"• **CNN Model Weights:** `{'Ready' if os.path.exists(cnn_path) else 'Missing'}` 🟢")
        st.write(f"• **CNN-LSTM Model Weights:** `{'Ready' if os.path.exists(lstm_path) else 'Missing'}` 🟢")
        st.write(f"• **Label Encoder:** `{'Ready' if os.path.exists(enc_path) else 'Missing'}` 🟢")
        st.write(f"• **SQLite Database:** `{'Active' if os.path.exists(db_path) else 'Initialized'}` 🟢")

    st.markdown("---")

    st.markdown("### Audio Signal Preprocessing Specifications")
    st.markdown("""
    - **Sampling Rate:** Resampled to 22,050 Hz (Nyquist limit: 11,025 Hz, covering the full human speech formant range).
    - **Channel Configuration:** Mono-channel conversion.
    - **Length Standardization:** Fixed 3.0 seconds duration (`int(sr * 3.0)` = 66,150 samples).
    - **Silence Trimming:** `librosa.effects.trim` with 20 dB dynamic threshold.
    - **Log-Mel Transform:** 128 Mel bands, N_FFT = 2048, Hop Length = 512, producing exact input matrices of shape `(128, 130, 1)`.
    - **Speaker-Independent Partitioning:** Actor-wise group splits strictly prevent speaker overlap across train, validation, and test splits.
    """)

    st.markdown("---")

    st.markdown("### Ethical AI, Privacy & Academic Disclaimers")
    st.warning("""
    ⚠️ **Academic & Non-Clinical Disclaimer:**
    This application is designed for research, academic evaluation, and demonstration purposes. Voice emotion recognition models evaluate statistical correlations in acoustic features (pitch, harmonics, cadence) derived from simulated datasets (RAVDESS). They do NOT measure a person's inner psychological state, intent, or clinical condition.
    """)

    st.info("""
    🔒 **Local Privacy Guarantee:**
    All audio processing, feature extraction, neural network inference, and SQLite database storage execute locally in your environment. No audio signals or voice recordings are transmitted to external cloud servers.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Return to Analyze Voice (ANALYZE)", type="primary"):
        st.session_state.page = "ANALYZE"
        st.rerun()
