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
import librosa
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT CALL)
# ---------------------------------------------------------
st.set_page_config(
    page_title="SONORA / EMOTIVA | AI Speech Emotion Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 2. BACKEND IMPORTS
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


def get_audio_mime(filename: str = None) -> str:
    """Return appropriate audio MIME type based on file extension."""
    if not filename:
        return 'audio/wav'
    ext = os.path.splitext(filename)[1].lower()
    return getattr(config, 'AUDIO_MIME_TYPES', {}).get(ext, 'audio/wav')


# ---------------------------------------------------------
# 3. CSS STYLING — SONORA EDITORIAL AUDIO-LAB DESIGN
# ---------------------------------------------------------
CSS_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

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

/* Global Typography & Background */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Streamlit Metric Overrides */
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
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
    padding: 2.5rem 1rem 1.5rem;
}
.hero-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 9999px;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: var(--accent-cyan);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 4.2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 20%, #93c5fd 60%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem;
}
.hero-subtitle {
    font-size: 1.25rem;
    font-weight: 400;
    color: var(--text-secondary);
    max-width: 680px;
    margin: 0 auto 2rem auto;
    line-height: 1.6;
}

/* Status Metric Cards */
.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0 2.5rem;
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
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.status-card-value {
    font-size: 1.35rem;
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
    padding: 1.5rem;
    background: rgba(12, 18, 32, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    margin: 2rem 0;
}
.pipeline-node {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    padding: 0.6rem 1.1rem;
    border-radius: 8px;
    font-size: 0.85rem;
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
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0 2rem;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
    border: 2px solid;
    position: relative;
}
.prediction-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1rem;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
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
    font-size: 0.9rem;
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
    'predictions_history': [],
    'analysis_results': None,
    'model_type': 'cnn_lstm',
    'timeline_results': None,
    'audio_features': None,
    'mfcc_n': 40,
    'spec_mels': 128,
    'trigger_analyze_now': False,
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


def execute_ser_pipeline(model_type: str, audio_data: np.ndarray, audio_sr: int, audio_name: str):
    """
    Executes the Speech Emotion Recognition inference pipeline,
    extracts features, computes timeline segments, and updates session state & history.
    Returns (success: bool, error_msg: str | None).
    """
    if audio_data is None or len(audio_data) == 0:
        return False, "No audio signal loaded to analyze."

    loaded_model, loaded_encoder, load_err = get_model_and_encoder(model_type)
    if loaded_model is None or loaded_encoder is None:
        return False, f"Model '{model_type.upper()}' is not trained yet. Run: python src/train.py --model {model_type}"

    try:
        if np.max(np.abs(audio_data)) > 0:
            audio_norm = librosa.util.normalize(audio_data)
        else:
            audio_norm = audio_data

        result = predict_emotion(audio_norm, loaded_model, loaded_encoder, sr=audio_sr)
        timeline = predict_segments(audio_norm, loaded_model, loaded_encoder, sr=audio_sr, min_duration_for_timeline=4.0)
        features = extract_audio_features(audio_norm, audio_sr)

        st.session_state.analysis_results = result
        st.session_state.timeline_results = timeline
        st.session_state.audio_features = features

        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": audio_name or "Voice Recording",
            "duration": round(len(audio_data) / audio_sr, 2),
            "model": model_type.upper(),
            "emotion": result['emotion'],
            "confidence": round(result['confidence'], 4),
            "confidence_pct": f"{result['confidence']*100:.1f}%",
            "top_3": ", ".join([f"{item['emotion']} ({item['confidence_pct']}%)" for item in result.get('top_3', [])])
        }
        st.session_state.predictions_history.insert(0, history_entry)
        return True, None
    except Exception as ex:
        return False, f"Inference pipeline failed: {str(ex)}\n{traceback.format_exc()}"

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

st.markdown("<hr style='border: 1px solid var(--border-subtle); margin-top: 0.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)

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

    c1, c2, c3 = st.columns([1, 1.2, 1])
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
            <div class='status-card-title'>Deep Architecture</div>
            <div class='status-card-value'>CNN-LSTM</div>
        </div>
        <div class='status-card'>
            <div class='status-card-title'>Audio DNA</div>
            <div class='status-card-value'>128 Mel Bands</div>
        </div>
        <div class='status-card'>
            <div class='status-card-title'>Emotion Classes</div>
            <div class='status-card-value'>8 Categories</div>
        </div>
        <div class='status-card'>
            <div class='status-card-title'>Standard Sample Rate</div>
            <div class='status-card-value'>22,050 Hz</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Architectural Pipeline Flow
    st.markdown("""
    <div style='text-align:center; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;'>
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
    st.markdown("<h4 style='text-align: center; margin: 2rem 0 1rem;'>Recognized Affective Classes</h4>", unsafe_allow_html=True)
    emo_cols_1 = st.columns(4)
    emo_cols_2 = st.columns(4)
    all_emo_cols = emo_cols_1 + emo_cols_2

    for idx, emotion in enumerate(config.EMOTIONS):
        col_color = config.EMOTION_COLORS.get(emotion, "#38bdf8")
        emoji = config.EMOTION_EMOJIS.get(emotion, "🎭")
        with all_emo_cols[idx]:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; border-top: 3px solid {col_color}; padding: 1rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.25rem;'>{emoji}</div>
                <div style='font-weight: 700; text-transform: capitalize;'>{emotion}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Quick Navigation Footer Actions
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
        if st.button("📜 Session History", use_container_width=True):
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
    st.markdown("<p style='color: var(--text-secondary);'>Upload, record, or test voice signals through deep acoustic models.</p>", unsafe_allow_html=True)

    # 1. Model Selection Control (Requirement 6)
    st.markdown("### 1. Select Model Architecture")
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        selected_model_type = st.radio(
            "Inference Architecture:",
            options=["cnn_lstm", "cnn"],
            format_func=lambda x: "CNN-LSTM (Main Spatiotemporal Network)" if x == "cnn_lstm" else "CNN Baseline (Spatial Spectrogram Network)",
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
            <div style='font-size: 0.8rem; color: var(--text-muted); text-transform:uppercase;'>Current Active Model</div>
            <div style='font-size: 1.15rem; font-weight:700; color:var(--text-primary);'>{active_model_name}</div>
            <div style='font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;'>
                { 'Captures both spectral patterns and temporal intonation across time.' if st.session_state.model_type == 'cnn_lstm' else 'Evaluates global time-frequency spectral power distributions.' }
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Audio Source Selection (Upload / Record / Sample Library)
    st.markdown("### 2. Audio Input")

    tab_upload, tab_record, tab_samples = st.tabs(["📤 Upload Audio (WAV / MP3)", "🎙️ Record Microphone (Live Audio)", "🎵 Demo Audio Library"])

    input_audio_bytes = None
    input_audio_name = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload audio file (WAV, MP3, FLAC, OGG, M4A)",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'],
            key="uploader_audio"
        )
        if uploaded_file is not None:
            input_audio_bytes = uploaded_file.read()
            input_audio_name = uploaded_file.name
            st.markdown(f"**Loaded File:** `{uploaded_file.name}`")
            
            up_col1, up_col2 = st.columns([1.5, 1])
            with up_col1:
                if st.button("⚡ ANALYZE UPLOADED AUDIO NOW", type="primary", key="btn_analyze_upload_direct", use_container_width=True):
                    y_up, sr_up = decode_audio_bytes(input_audio_bytes, target_sr=config.SR)
                    if len(y_up) > 0:
                        if np.max(np.abs(y_up)) > 0:
                            y_up = librosa.util.normalize(y_up)
                        st.session_state.audio_bytes = input_audio_bytes
                        st.session_state.audio_name = input_audio_name
                        st.session_state.audio_data = y_up
                        st.session_state.audio_sr = sr_up
                        st.session_state.audio_duration = float(len(y_up) / sr_up)
                        with st.spinner("Classifying vocal affect..."):
                            ok, err = execute_ser_pipeline(st.session_state.model_type, y_up, sr_up, input_audio_name)
                            if ok:
                                st.rerun()
                            else:
                                st.error(err)
            with up_col2:
                if st.button("🎛️ Open in Audio Lab", key="btn_up_audio_lab", use_container_width=True):
                    y_up, sr_up = decode_audio_bytes(input_audio_bytes, target_sr=config.SR)
                    if len(y_up) > 0:
                        st.session_state.audio_bytes = input_audio_bytes
                        st.session_state.audio_name = input_audio_name
                        st.session_state.audio_data = y_up
                        st.session_state.audio_sr = sr_up
                        st.session_state.audio_duration = float(len(y_up) / sr_up)
                        st.session_state.page = "AUDIO LAB"
                        st.rerun()
        else:
            st.button("⚡ ANALYZE UPLOADED AUDIO (Upload an audio file above to enable)", disabled=True, key="btn_up_disabled", use_container_width=True)

    with tab_record:
        st.markdown("#### 🎙️ Live Voice Recording & Real-Time Emotion Classification")
        st.markdown(
            "<p style='color: var(--text-secondary); font-size: 0.92rem; margin-bottom: 12px;'>"
            "<strong>How to use:</strong><br>"
            "1. Click the microphone button below to start recording.<br>"
            "2. Speak into your microphone with emotional intonation (3 to 5 seconds recommended).<br>"
            "3. Click the stop button to finish recording.<br>"
            "4. Click <strong>⚡ ANALYZE RECORDED VOICE NOW</strong> to classify emotion."
            "</p>",
            unsafe_allow_html=True
        )
        if hasattr(st, 'audio_input'):
            recorded_audio = st.audio_input("Record voice sample from your microphone", key="mic_recorder")
            if recorded_audio is not None:
                raw_mic_bytes = recorded_audio.getvalue()
                if raw_mic_bytes and len(raw_mic_bytes) > 0:
                    y_mic, sr_mic = decode_audio_bytes(raw_mic_bytes, target_sr=config.SR)
                    if len(y_mic) > 0:
                        dur_m = float(len(y_mic) / sr_mic)
                        clean_wav = audio_to_wav_bytes(y_mic, sr_mic)
                        input_audio_bytes = clean_wav
                        input_audio_name = "Live Microphone Recording"
                        
                        # Auto-update active audio state
                        st.session_state.audio_bytes = clean_wav
                        st.session_state.audio_name = "Live Microphone Recording"
                        st.session_state.audio_data = y_mic
                        st.session_state.audio_sr = sr_mic
                        st.session_state.audio_duration = dur_m
                        
                        st.success(f"✅ Voice recorded: **{dur_m:.2f} seconds** ({sr_mic} Hz)")
                        st.audio(clean_wav, format='audio/wav')
                        
                        rec_btn_c1, rec_btn_c2 = st.columns([1.5, 1])
                        with rec_btn_c1:
                            if st.button("⚡ ANALYZE RECORDED VOICE NOW", type="primary", key="btn_analyze_mic_direct", use_container_width=True):
                                with st.spinner(f"Analyzing live voice with {st.session_state.model_type.upper()}..."):
                                    ok, err = execute_ser_pipeline(st.session_state.model_type, y_mic, sr_mic, "Live Microphone Recording")
                                    if ok:
                                        st.rerun()
                                    else:
                                        st.error(err)
                        with rec_btn_c2:
                            if st.button("🎛️ Open in Audio Lab", key="btn_mic_audio_lab", use_container_width=True):
                                st.session_state.page = "AUDIO LAB"
                                st.rerun()
                    else:
                        st.warning("⚠️ Microphone captured 0 decodable audio frames. Please check microphone permissions and speak louder.")
                        st.button("⚡ ANALYZE RECORDED VOICE (No audio detected)", disabled=True, key="btn_mic_empty_disabled", use_container_width=True)
            else:
                st.info("🎙️ Speak into your microphone and click stop recording above. Once recorded, the **⚡ Analyze Live Voice** button will activate.")
                st.button("⚡ ANALYZE RECORDED VOICE (Record your voice above to enable)", disabled=True, key="btn_mic_disabled", use_container_width=True)
        else:
            st.info("Browser microphone recording requires Streamlit >= 1.39. Please upload an audio file.")

    with tab_samples:
        st.write("Select a pre-synthesized acoustic demo voice representing each of the 8 emotions (or multi-emotion speech):")
        demo_fmt = st.radio("Demo Format:", ["WAV", "MP3"], horizontal=True, key="demo_audio_format")
        sample_ext = ".mp3" if demo_fmt == "MP3" else ".wav"
        samples_dir = os.path.join(config.PROJECT_ROOT, "samples")
        
        # Row 1: First 4 emotions
        row1_cols = st.columns(4)
        row1_samples = [
            ("😐 Neutral (3.2s)", f"demo_neutral{sample_ext}"),
            ("😌 Calm (3.2s)", f"demo_calm{sample_ext}"),
            ("😊 Happy (3.2s)", f"demo_happy{sample_ext}"),
            ("😢 Sad (3.2s)", f"demo_sad{sample_ext}"),
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

        # Row 2: Next 4 emotions
        row2_cols = st.columns(4)
        row2_samples = [
            ("😡 Angry (3.2s)", f"demo_angry{sample_ext}"),
            ("😨 Fearful (3.2s)", f"demo_fearful{sample_ext}"),
            ("🤢 Disgust (3.2s)", f"demo_disgust{sample_ext}"),
            ("😲 Surprised (3.2s)", f"demo_surprised{sample_ext}"),
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

        # Row 3: Multi-Emotion Long Audio for Timeline Analysis
        long_path = os.path.join(samples_dir, f"sample_long_speech{sample_ext}")
        if st.button(f"⏱️ Long Speech (6.5s — Calm + Happy in {demo_fmt})", key="btn_demo_long_speech", use_container_width=True):
            if os.path.exists(long_path):
                with open(long_path, 'rb') as f:
                    input_audio_bytes = f.read()
                    input_audio_name = f"sample_long_speech{sample_ext}"

    # Load into session state if new audio is provided
    if input_audio_bytes is not None and (st.session_state.audio_bytes != input_audio_bytes):
        # 1. Attempt universal in-memory decoding
        y_decoded, sr_decoded = decode_audio_bytes(input_audio_bytes, target_sr=config.SR)
        
        if len(y_decoded) > 0:
            dur_sec = float(len(y_decoded) / sr_decoded)
            if dur_sec < 0.5:
                st.error(f"❌ Invalid Audio: Audio is too short ({dur_sec:.2f}s). Minimum required duration is 0.5s.")
            elif dur_sec > 120.0:
                st.error(f"❌ Invalid Audio: Audio is too long ({dur_sec:.1f}s). Maximum allowed duration is 120.0s.")
            elif np.isnan(y_decoded).any() or np.isinf(y_decoded).any():
                st.error("❌ Invalid Audio: Audio contains corrupted numerical data (NaN or Inf values).")
            else:
                if np.max(np.abs(y_decoded)) > 0:
                    y_decoded = librosa.util.normalize(y_decoded)

                # Ensure clean audio bytes for browser playback
                if input_audio_name and input_audio_name.endswith('.wav'):
                    clean_playback_bytes = audio_to_wav_bytes(y_decoded, sr_decoded)
                else:
                    clean_playback_bytes = input_audio_bytes

                st.session_state.audio_bytes = clean_playback_bytes
                st.session_state.audio_name = input_audio_name
                st.session_state.audio_data = y_decoded
                st.session_state.audio_sr = sr_decoded
                st.session_state.audio_duration = dur_sec
                # Reset previous analysis for the new audio
                st.session_state.analysis_results = None
                st.session_state.timeline_results = None
                st.session_state.audio_features = None
                st.rerun()
        else:
            # Fallback file validation for edge cases / tests
            file_ext = os.path.splitext(input_audio_name)[1].lower() if input_audio_name else '.wav'
            if not file_ext or file_ext not in getattr(config, 'SUPPORTED_AUDIO_EXTENSIONS', ['.wav', '.mp3']):
                file_ext = '.wav'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(input_audio_bytes)
                tmp_path = tmp.name

            try:
                is_valid, val_msg, dur_sec, file_sr = validate_audio_file(tmp_path)
                if not is_valid:
                    st.error(f"❌ Invalid Audio: {val_msg}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # 3. Audio Preview & Waveform
    if st.session_state.audio_data is not None:
        st.markdown("### 3. Audio Preview & Acoustics")
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.write(f"**Loaded Signal:** `{st.session_state.audio_name}`")
            st.audio(st.session_state.audio_bytes, format=get_audio_mime(st.session_state.audio_name))

            dur_sec = len(st.session_state.audio_data) / st.session_state.audio_sr
            stat_c1, stat_c2 = st.columns(2)
            stat_c1.metric("Duration", f"{dur_sec:.2f} s")
            stat_c2.metric("Sample Rate", f"{st.session_state.audio_sr} Hz")

            # Reset Button (Requirement 10)
            if st.button("🔄 RESET ANALYSIS", use_container_width=True):
                st.session_state.audio_data = None
                st.session_state.audio_sr = None
                st.session_state.audio_name = None
                st.session_state.audio_bytes = None
                st.session_state.audio_duration = None
                st.session_state.analysis_results = None
                st.session_state.timeline_results = None
                st.session_state.audio_features = None
                st.rerun()

            # Cross-page Navigation to Audio Lab (Requirement 3)
            if st.button("🎛️ Open in Audio Lab →", use_container_width=True):
                st.session_state.page = "AUDIO LAB"
                st.rerun()

        with p_col2:
            st.plotly_chart(
                plot_waveform(st.session_state.audio_data, st.session_state.audio_sr),
                use_container_width=True
            )

        # 4. Trigger Analysis
        st.markdown("---")
        action_col1, action_col2, action_col3 = st.columns([1, 1.5, 1])
        with action_col2:
            analyze_clicked = st.button(
                f"🔬 ANALYZE VOICE AFFECT ({st.session_state.model_type.upper()})",
                type="primary",
                key="btn_main_analyze_trigger",
                use_container_width=True
            )

        if analyze_clicked:
            with st.spinner(f"Extracting acoustic features and inferring affect with {st.session_state.model_type.upper()}..."):
                ok, err = execute_ser_pipeline(
                    st.session_state.model_type,
                    st.session_state.audio_data,
                    st.session_state.audio_sr,
                    st.session_state.audio_name
                )
                if ok:
                    st.rerun()
                else:
                    st.error(f"## Inference Error\n{err}")

    # 5. Display Analysis Results
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        primary_emo = res['emotion']
        conf = res['confidence']
        conf_pct = res['confidence_pct']
        conf_lvl = res['confidence_level']
        emo_color = config.EMOTION_COLORS.get(primary_emo, "#38bdf8")
        emo_emoji = config.EMOTION_EMOJIS.get(primary_emo, "🎭")

        # Result Hero Banner
        badge_bg = "rgba(16, 185, 129, 0.2)" if conf_lvl == "High" else "rgba(245, 158, 11, 0.2)" if conf_lvl == "Moderate" else "rgba(239, 68, 68, 0.2)"
        badge_color = "#10b981" if conf_lvl == "High" else "#f59e0b" if conf_lvl == "Moderate" else "#ef4444"

        st.markdown(f"""
        <div class='prediction-box' style='border-color: {emo_color};'>
            <div style='font-size: 3.5rem;'>{emo_emoji}</div>
            <div style='font-size: 2.6rem; font-weight:800; color:{emo_color}; text-transform:uppercase; margin-top:0.25rem;'>
                {primary_emo}
            </div>
            <div style='font-size: 1.25rem; color: var(--text-secondary); margin-top:0.5rem;'>
                Inference Confidence: <strong style='color:#f8fafc;'>{conf_pct}%</strong>
            </div>
            <div class='prediction-badge' style='background:{badge_bg}; color:{badge_color}; border: 1px solid {badge_color};'>
                {conf_lvl} Confidence Level
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("ℹ️ **Confidence Interpretation:** Reflects model probability distribution over the 8 trained RAVDESS classes. It represents acoustic feature affinity, not an absolute diagnosis of human emotion.")

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

        # Emotion Timeline (Requirement 8)
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
            st.info("Audio is too short for segment-level emotion analysis.")

        # Voice Energy (RMS)
        st.markdown("---")
        st.markdown("### ⚡ Voice Energy Dynamics (RMS)")
        if st.session_state.audio_data is not None:
            st.plotly_chart(
                plot_rms_energy(st.session_state.audio_data, st.session_state.audio_sr),
                use_container_width=True
            )

        # Audio Signal Insights Expander
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

        # Why This Result Expander
        with st.expander("🤔 Acoustic Interpretation & Emotion Markers", expanded=False):
            desc = config.EMOTION_DESCRIPTIONS.get(primary_emo, "No description available.")
            st.markdown(f"**Acoustic Profile for {primary_emo.capitalize()}:**")
            st.write(desc)
            st.caption("Based on supervised learning from professional actors in the RAVDESS corpus.")

        # Export Analysis Report Button (Requirement 11)
        st.markdown("---")
        html_report_content = generate_html_report(
            res,
            st.session_state.audio_name,
            st.session_state.audio_sr,
            st.session_state.audio_data,
            active_model_name,
            st.session_state.audio_features
        )
        report_filename = f"sonora_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        exp_c1, exp_c2, exp_c3 = st.columns([1, 1.5, 1])
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
# PAGE 3: AUDIO LAB
# =========================================================
elif st.session_state.page == "AUDIO LAB":
    st.markdown("<h2>🎛️ Audio Signal Processing Workstation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary);'>Interactive digital signal processing, filterbanks, MFCCs, and spectral distributions.</p>", unsafe_allow_html=True)

    if st.session_state.audio_data is None:
        st.info("💡 No active audio signal loaded. Upload or record audio below to begin acoustic inspection:")
        
        lab_up_tab1, lab_up_tab2, lab_up_tab3 = st.tabs(["📤 Upload Audio", "🎙️ Record Microphone", "🎵 Demo Audio Library"])
        
        lab_audio_bytes = None
        lab_audio_name = None
        
        with lab_up_tab1:
            lab_up_file = st.file_uploader(
                "Upload audio file (WAV, MP3, FLAC, OGG, M4A)",
                type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'],
                key="lab_uploader_audio"
            )
            if lab_up_file is not None:
                lab_audio_bytes = lab_up_file.read()
                lab_audio_name = lab_up_file.name
                
        with lab_up_tab2:
            st.markdown("#### 🎙️ Live Voice Recording")
            if hasattr(st, 'audio_input'):
                lab_rec = st.audio_input("Record voice sample from microphone", key="lab_mic_recorder")
                if lab_rec is not None:
                    raw_bytes = lab_rec.getvalue()
                    if raw_bytes and len(raw_bytes) > 0:
                        y_m, sr_m = decode_audio_bytes(raw_bytes, target_sr=config.SR)
                        if len(y_m) > 0:
                            dur_lab = float(len(y_m) / sr_m)
                            lab_audio_bytes = audio_to_wav_bytes(y_m, sr_m)
                            lab_audio_name = "Live Microphone Recording"
                            st.success(f"✅ Voice recorded: **{dur_lab:.2f} s** ({sr_m} Hz)")
                            st.audio(lab_audio_bytes, format='audio/wav')
                            if st.button("⚡ ANALYZE RECORDING IN AUDIO LAB", type="primary", key="btn_lab_mic_analyze", use_container_width=True):
                                if np.max(np.abs(y_m)) > 0:
                                    y_m = librosa.util.normalize(y_m)
                                st.session_state.audio_bytes = lab_audio_bytes
                                st.session_state.audio_name = lab_audio_name
                                st.session_state.audio_data = y_m
                                st.session_state.audio_sr = sr_m
                                st.session_state.audio_duration = dur_lab
                                with st.spinner("Analyzing audio in Audio Lab..."):
                                    execute_ser_pipeline(st.session_state.model_type, y_m, sr_m, lab_audio_name)
                                st.rerun()
                        else:
                            st.warning("⚠️ Microphone captured 0 decodable audio frames.")
                else:
                    st.info("🎙️ Speak into microphone and stop recording. Click the **⚡ Analyze Recording** button to load.")
                    st.button("⚡ ANALYZE RECORDING (Record voice first to enable)", disabled=True, key="btn_lab_mic_disabled", use_container_width=True)
            else:
                st.info("Browser microphone recording requires Streamlit >= 1.39.")
                
        with lab_up_tab3:
            st.write("Pick a pre-synthesized acoustic demo:")
            lab_demo_cols = st.columns(4)
            demo_choices = [
                ("😊 Happy", "demo_happy.wav"),
                ("😢 Sad", "demo_sad.wav"),
                ("😡 Angry", "demo_angry.wav"),
                ("😌 Calm", "demo_calm.wav"),
            ]
            samples_dir = os.path.join(config.PROJECT_ROOT, "samples")
            for idx, (label, sfile) in enumerate(demo_choices):
                with lab_demo_cols[idx]:
                    if st.button(label, key=f"btn_lab_demo_{sfile}", use_container_width=True):
                        s_path = os.path.join(samples_dir, sfile)
                        if not os.path.exists(s_path):
                            alt = sfile.replace("demo_", "sample_")
                            s_path = os.path.join(samples_dir, alt) if os.path.exists(os.path.join(samples_dir, alt)) else s_path
                        if os.path.exists(s_path):
                            with open(s_path, 'rb') as f:
                                lab_audio_bytes = f.read()
                                lab_audio_name = sfile
                                
        if lab_audio_bytes is not None:
            y_dec, sr_dec = decode_audio_bytes(lab_audio_bytes, target_sr=config.SR)
            if len(y_dec) > 0:
                if np.max(np.abs(y_dec)) > 0:
                    y_dec = librosa.util.normalize(y_dec)
                st.session_state.audio_bytes = lab_audio_bytes
                st.session_state.audio_name = lab_audio_name
                st.session_state.audio_data = y_dec
                st.session_state.audio_sr = sr_dec
                st.session_state.audio_duration = float(len(y_dec) / sr_dec)
                st.rerun()

        st.markdown("---")
        if st.button("← Switch to Main ANALYZE Dashboard", type="secondary"):
            st.session_state.page = "ANALYZE"
            st.rerun()
    else:
        y = st.session_state.audio_data
        sr = st.session_state.audio_sr

        # Top Audio Header & Controls
        top_col1, top_col2, top_col3 = st.columns([2, 1, 1.2])
        with top_col1:
            st.write(f"**Active Signal:** `{st.session_state.audio_name}`")
            st.audio(st.session_state.audio_bytes, format=get_audio_mime(st.session_state.audio_name))
        with top_col2:
            st.metric("Total Samples", f"{len(y):,}")
            st.metric("Duration", f"{len(y)/sr:.2f} s")
        with top_col3:
            st.metric("Sample Rate", f"{sr} Hz")
            
            # Action Buttons: Analyze Button for Audio Lab Page
            btn_act1, btn_act2 = st.columns(2)
            with btn_act1:
                run_lab_analyze = st.button("⚡ ANALYZE", type="primary", use_container_width=True, help="Run deep emotion classification on this audio")
            with btn_act2:
                if st.button("📋 Full Report", use_container_width=True, help="Open full emotion results on Analyze page"):
                    st.session_state.page = "ANALYZE"
                    st.rerun()

        # Handle direct Analyze execution from Audio Lab
        if 'run_lab_analyze' in locals() and run_lab_analyze:
            with st.spinner("Classifying vocal affect..."):
                ok, err = execute_ser_pipeline(
                    st.session_state.model_type,
                    st.session_state.audio_data,
                    st.session_state.audio_sr,
                    st.session_state.audio_name
                )
                if ok:
                    st.success(f"Affect Classified: **{st.session_state.analysis_results['emotion'].upper()}** ({st.session_state.analysis_results['confidence']*100:.1f}% confidence)")
                else:
                    st.error(err)

        # If analysis results are already computed, display a compact summary banner in Audio Lab
        if st.session_state.analysis_results is not None:
            res_summary = st.session_state.analysis_results
            p_emo = res_summary['emotion']
            p_conf = res_summary['confidence'] * 100
            p_color = config.EMOTION_COLORS.get(p_emo, '#6366f1')
            p_emoji = config.EMOTION_EMOJIS.get(p_emo, '🎭')
            
            st.markdown(f"""
            <div style='background: rgba(18, 25, 44, 0.7); border: 1px solid rgba(255,255,255,0.1); border-left: 4px solid {p_color}; border-radius: 10px; padding: 12px 18px; margin: 12px 0; display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <span style='font-size: 1.4rem; margin-right: 8px;'>{p_emoji}</span>
                    <strong style='font-size: 1.1rem; color: #fff;'>Predicted Affect: <span style='color:{p_color}; text-transform: uppercase;'>{p_emo}</span></strong>
                    <span style='color: var(--text-secondary); margin-left: 12px;'>Confidence: <strong>{p_conf:.1f}%</strong> ({res_summary.get('confidence_level', 'High')})</span>
                </div>
                <div>
                    <span style='font-size: 0.85rem; color: var(--accent-cyan);'>Model: {st.session_state.model_type.upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        lab_tabs = st.tabs([
            "Waveform", "Mel-Spectrogram", "MFCC Coefficients",
            "Spectral Features", "RMS Loudness", "Zero Crossing Rate"
        ])

        # TAB 1: WAVEFORM
        with lab_tabs[0]:
            st.markdown("#### Time-Domain Signal Amplitude")
            st.write("Displays the instantaneous raw acoustic pressure amplitude normalized to [-1.0, 1.0].")
            st.plotly_chart(plot_waveform(y, sr), use_container_width=True)

        # TAB 2: MEL-SPECTROGRAM (Requirement 20: Interactive Controls)
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

        # TAB 3: MFCC (Requirement 20: Dynamic Coefficient Slider)
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


# =========================================================
# PAGE 4: INSIGHTS
# =========================================================
elif st.session_state.page == "INSIGHTS":
    st.markdown("<h2>📊 RAVDESS Dataset Insights & Affect Statistics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary);'>Acoustic corpus composition and emotion class distributions.</p>", unsafe_allow_html=True)

    df_meta = load_ravdess_metadata(config.DATASET_PATH)

    # Missing Dataset Handling (Requirement 12)
    if df_meta.empty or not os.path.exists(config.DATASET_PATH):
        st.error("## DATASET NOT FOUND")
        st.markdown(f"""
        The RAVDESS audio dataset was not detected in:
        `{config.DATASET_PATH}`

        ### How to Prepare the Dataset:
        1. Download the **Audio_Speech_Actors_01-24.zip** archive from Zenodo:
           [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)
        2. Extract the folders `Actor_01`, `Actor_02`, ..., `Actor_24` into:
           `dataset/RAVDESS/`
        3. Or run the automated dataset generator for local testing:
           ```bash
           python src/create_sample_dataset.py
           ```
        4. Once placed, train the deep models:
           ```bash
           python src/train.py --model cnn_lstm
           python src/train.py --model cnn
           ```
        """)
    else:
        # Real Dataset Statistics (Requirement 7 & 18)
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Total Audio Recordings", f"{len(df_meta):,}")
        ic2.metric("Professional Actors", df_meta['actor'].nunique())
        ic3.metric("Emotion Categories", df_meta['emotion'].nunique())
        ic4.metric("Standard Duration", f"{config.DURATION:.1f} s")

        st.markdown("### Emotion Distribution in Dataset")
        st.plotly_chart(plot_emotion_distribution(df_meta), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎭 Interactive Emotion Acoustic Profile")

    selected_emo = st.selectbox("Select Affect Category:", config.EMOTIONS, key="select_insight_emo")

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
        <p style='font-size: 1.05rem; margin-top: 1rem; color: var(--text-primary); line-height: 1.6;'>
            {emo_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Cross-Navigation (Requirement 3: INSIGHTS -> MODELS)
    if st.button("Explore Model Benchmarks & Architecture (MODELS) →", type="primary"):
        st.session_state.page = "MODELS"
        st.rerun()


# =========================================================
# PAGE 5: MODELS
# =========================================================
elif st.session_state.page == "MODELS":
    st.markdown("<h2>🧪 Deep Learning Models & Benchmark Evaluations</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary);'>Comparative performance metrics, confusion matrices, and training convergence curves.</p>", unsafe_allow_html=True)

    # Check for evaluation results
    comp_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    eval_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    cnn_hist_path = os.path.join(config.RESULTS_DIR, "cnn_history.json")
    lstm_hist_path = os.path.join(config.RESULTS_DIR, "cnn_lstm_history.json")

    has_comparison = os.path.exists(comp_path)
    has_evaluation = os.path.exists(eval_path)
    has_history = os.path.exists(lstm_hist_path) or os.path.exists(cnn_hist_path)

    # 1. Benchmark Comparison Table & Graph
    st.markdown("### 1. Comparative Architecture Performance")
    if has_comparison:
        comp_df = pd.read_csv(comp_path)
        st.plotly_chart(plot_model_comparison(comp_df), use_container_width=True)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Model comparison metrics not available yet. Please train and evaluate models.")
        st.code("python src/evaluate.py", language="bash")

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
            st.info("CNN-LSTM history not available yet.")

    with hist_col2:
        st.markdown("#### CNN Baseline Curves")
        if os.path.exists(cnn_hist_path):
            with open(cnn_hist_path, 'r') as f:
                cnn_hist = json.load(f)
            st.plotly_chart(plot_training_curves(cnn_hist), use_container_width=True)
        else:
            st.info("CNN Baseline history not available yet.")

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
                st.markdown("#### Classification Report")
                st.code(selected_eval['report'])
    else:
        st.info("Evaluation results not available yet. Run `python src/evaluate.py`.")

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
        ├── MaxPooling2D(2, 2) + Dropout(0.3)
        │
        ├── Conv2D(64, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 4) + Dropout(0.3)
        │
        ├── Permute((2, 1, 3)) [Time axis first]
        ├── Reshape((16 TimeSteps, 2048 Features))
        │
        ├── LSTM(128 Units) + Dropout(0.4)
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
        ├── MaxPooling2D(2, 2) + Dropout(0.3)
        │
        ├── Conv2D(64, 3x3, same) + BatchNorm + ReLU
        ├── MaxPooling2D(2, 2) + Dropout(0.3)
        │
        ├── GlobalAveragePooling2D()
        ├── Dense(64, ReLU) + Dropout(0.3)
        └── Dense(8, Softmax) -> Categorical Affect
        ```
        """)


# =========================================================
# PAGE 6: HISTORY
# =========================================================
elif st.session_state.page == "HISTORY":
    st.markdown("<h2>📜 Session Inference History & Audit Log</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary);'>Audit trail of all audio predictions generated during this session.</p>", unsafe_allow_html=True)

    if not st.session_state.predictions_history:
        st.info("No audio predictions performed yet in this session. Go to ANALYZE to run voice classification.")
        if st.button("← Go to Analyze"):
            st.session_state.page = "ANALYZE"
            st.rerun()
    else:
        df_hist = pd.DataFrame(st.session_state.predictions_history)

        # Filtering Controls (Requirement 9)
        st.markdown("### Filter & Sort History")
        fil_c1, fil_c2, fil_c3 = st.columns(3)

        with fil_c1:
            all_emotions_in_hist = ["ALL"] + sorted(list(df_hist['emotion'].unique()))
            filter_emotion = st.selectbox("Filter by Emotion:", all_emotions_in_hist, key="fil_emo")

        with fil_c2:
            all_models_in_hist = ["ALL"] + sorted(list(df_hist['model'].unique()))
            filter_model = st.selectbox("Filter by Model:", all_models_in_hist, key="fil_mod")

        with fil_c3:
            sort_by = st.selectbox(
                "Sort Table By:",
                ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence", "Duration"],
                key="sort_hist"
            )

        # Apply Filters
        filtered_df = df_hist.copy()
        if filter_emotion != "ALL":
            filtered_df = filtered_df[filtered_df['emotion'] == filter_emotion]
        if filter_model != "ALL":
            filtered_df = filtered_df[filtered_df['model'] == filter_model]

        # Apply Sorting
        if sort_by == "Newest First":
            filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        elif sort_by == "Oldest First":
            filtered_df = filtered_df.sort_values('timestamp', ascending=True)
        elif sort_by == "Highest Confidence":
            filtered_df = filtered_df.sort_values('confidence', ascending=False)
        elif sort_by == "Lowest Confidence":
            filtered_df = filtered_df.sort_values('confidence', ascending=True)
        elif sort_by == "Duration":
            filtered_df = filtered_df.sort_values('duration', ascending=False)

        st.write(f"Showing **{len(filtered_df)}** of **{len(df_hist)}** predictions.")
        st.dataframe(filtered_df, use_container_width=True)

        # Action Buttons: CSV Export (Requirement 11) & Clear (Requirement 9)
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 EXPORT HISTORY (CSV)",
                data=csv_data,
                file_name=f"sonora_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with act_c2:
            if st.button("🗑️ CLEAR SESSION HISTORY", use_container_width=True):
                st.session_state.predictions_history = []
                st.rerun()

        # Session Emotion Distribution Chart
        if len(df_hist) >= 2:
            st.markdown("---")
            st.markdown("### Session Emotion Distribution")
            emo_counts = df_hist['emotion'].value_counts().reset_index()
            emo_counts.columns = ['emotion', 'count']
            fig_hist = px.bar(
                emo_counts, x='emotion', y='count',
                color='emotion',
                color_discrete_map=config.EMOTION_COLORS,
                template='plotly_dark',
                title="Affect Frequency in Current Session"
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("About System Architecture & Methodology →"):
            st.session_state.page = "ABOUT"
            st.rerun()


# =========================================================
# PAGE 7: ABOUT
# =========================================================
elif st.session_state.page == "ABOUT":
    st.markdown("<h2>ℹ️ About SONORA / EMOTIVA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary);'>Architectural specifications, acoustic signal methodology, and ethical AI frameworks.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card' style='margin-bottom: 2rem;'>
        <h3 style='color: var(--accent-cyan); margin-top: 0;'>Project Abstract</h3>
        <p style='line-height: 1.7; font-size: 1.05rem;'>
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
            <div style='font-size: 1.8rem;'>🧠</div>
            <strong>TensorFlow 2.x / Keras</strong><br>
            <small style='color:var(--text-muted);'>Deep Spatiotemporal Models</small>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size: 1.8rem;'>🎵</div>
            <strong>Librosa / SoundFile</strong><br>
            <small style='color:var(--text-muted);'>Acoustic DSP Pipeline</small>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size: 1.8rem;'>📊</div>
            <strong>Plotly & Matplotlib</strong><br>
            <small style='color:var(--text-muted);'>Interactive Audio Visualizations</small>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class='glass-card' style='text-align:center;'>
            <div style='font-size: 1.8rem;'>🎈</div>
            <strong>Streamlit</strong><br>
            <small style='color:var(--text-muted);'>High-Performance Web Client</small>
        </div>
        """, unsafe_allow_html=True)

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
    All audio processing, feature extraction, and neural network inference execute locally in your execution environment. No audio signals or voice recordings are transmitted to external servers.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Return to Analyze Voice (ANALYZE)", type="primary"):
        st.session_state.page = "ANALYZE"
        st.rerun()
