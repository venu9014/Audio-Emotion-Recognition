"""
EMOTIVA — Report Generation Module
Generates comprehensive self-contained HTML analysis reports with embedded
base64 visualizations, acoustic features, and model metadata.
"""
import io
import base64
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa

import config


def generate_waveform_base64(y: np.ndarray, sr: int) -> str:
    """Generate dark-themed waveform PNG encoded as base64."""
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#0d1322')
    ax.set_facecolor('#0d1322')
    t = np.linspace(0, len(y) / sr, len(y))
    ax.plot(t, y, color='#06b6d4', linewidth=0.8)
    ax.set_title("Audio Waveform", color='#e2e8f0', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Time (s)", color='#94a3b8', fontsize=9)
    ax.set_ylabel("Amplitude", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.5)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_spectrogram_base64(y: np.ndarray, sr: int) -> str:
    """Generate dark-themed Mel-Spectrogram PNG encoded as base64."""
    fig, ax = plt.subplots(figsize=(8, 3.0), facecolor='#0d1322')
    ax.set_facecolor('#0d1322')

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )
    ref_val = np.max(S) if np.max(S) > 0 else 1.0
    S_dB = librosa.power_to_db(S, ref=ref_val)

    img = ax.imshow(S_dB, aspect='auto', origin='lower', cmap='viridis',
                    extent=[0, len(y)/sr, 0, sr/2])
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Power (dB)', color='#94a3b8', fontsize=9)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=8)

    ax.set_title("Log-Mel Spectrogram (Audio DNA)", color='#e2e8f0', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Time (s)", color='#94a3b8', fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_html_report(
    analysis_results: dict,
    audio_name: str,
    audio_sr: int,
    audio_data: np.ndarray,
    model_name: str,
    features: dict = None
) -> str:
    """
    Generate complete self-contained HTML analysis report.
    """
    res = analysis_results
    emotion = res.get('emotion', 'unknown').capitalize()
    raw_emotion = res.get('emotion', 'neutral').lower()
    confidence = res.get('confidence', 0.0)
    confidence_pct = f"{confidence * 100:.1f}%"
    confidence_level = res.get('confidence_level', 'Moderate')
    color = config.EMOTION_COLORS.get(raw_emotion, '#3b82f6')
    emoji = config.EMOTION_EMOJIS.get(raw_emotion, '🎭')
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    duration_sec = len(audio_data) / audio_sr if (audio_data is not None and len(audio_data) > 0) else 0.0

    # Generate charts
    waveform_b64 = generate_waveform_base64(audio_data, audio_sr) if audio_data is not None else ""
    spectrogram_b64 = generate_spectrogram_base64(audio_data, audio_sr) if audio_data is not None else ""

    # Top-3 list HTML
    sorted_probs = sorted(res.get('probabilities', {}).items(), key=lambda x: x[1], reverse=True)
    top_3_html = ""
    for rank, (emo, prob) in enumerate(sorted_probs[:3], start=1):
        emo_col = config.EMOTION_COLORS.get(emo, '#cbd5e1')
        emo_emo = config.EMOTION_EMOJIS.get(emo, '•')
        top_3_html += f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding: 8px 12px; margin-bottom: 6px; background:#131b2e; border-radius:6px; border-left: 4px solid {emo_col};">
            <span style="font-weight:600; text-transform:capitalize;">{rank}. {emo_emo} {emo}</span>
            <span style="font-weight:700; color:{emo_col};">{prob*100:.1f}%</span>
        </div>
        """

    # Audio features table HTML
    features_html = ""
    if features:
        features_html = f"""
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 15px;">
            <div style="background:#131b2e; padding:10px; border-radius:6px; text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8;">RMS ENERGY</div>
                <div style="font-size:1.1rem; font-weight:700; color:#38bdf8;">{features.get('rms_mean', 0):.4f}</div>
            </div>
            <div style="background:#131b2e; padding:10px; border-radius:6px; text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8;">ZERO CROSSING</div>
                <div style="font-size:1.1rem; font-weight:700; color:#38bdf8;">{features.get('zcr_mean', 0):.4f}</div>
            </div>
            <div style="background:#131b2e; padding:10px; border-radius:6px; text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8;">SPECTRAL CENTROID</div>
                <div style="font-size:1.1rem; font-weight:700; color:#38bdf8;">{features.get('spectral_centroid_mean', 0):.0f} Hz</div>
            </div>
            <div style="background:#131b2e; padding:10px; border-radius:6px; text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8;">SPECTRAL BANDWIDTH</div>
                <div style="font-size:1.1rem; font-weight:700; color:#38bdf8;">{features.get('spectral_bandwidth_mean', 0):.0f} Hz</div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EMOTIVA Analysis Report — {audio_name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
            background: #080c14;
            color: #e2e8f0;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .report-card {{
            max-width: 860px;
            margin: 0 auto;
            background: #0e1526;
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 36px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .brand {{
            font-size: 1.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .prediction-hero {{
            background: #131b2e;
            border: 2px solid {color}40;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        }}
        .emotion-title {{
            font-size: 2.4rem;
            font-weight: 800;
            color: {color};
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 24px 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .meta-item {{
            background: #131b2e;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #1e293b;
        }}
        .meta-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }}
        .meta-val {{ font-size: 0.95rem; font-weight: 600; color: #f8fafc; margin-top: 2px; }}
        .chart-box {{
            background: #0d1322;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
            text-align: center;
        }}
        .chart-box img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .disclaimer {{
            margin-top: 32px;
            padding: 16px;
            border-radius: 8px;
            background: rgba(239, 68, 68, 0.08);
            border-left: 4px solid #ef4444;
            font-size: 0.8rem;
            color: #cbd5e1;
            line-height: 1.6;
        }}
        .footer {{
            margin-top: 24px;
            text-align: center;
            font-size: 0.75rem;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <div class="brand">EMOTIVA / SONORA</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:2px;">AI Speech Emotion Intelligence Laboratory</div>
            </div>
            <div style="text-align:right;">
                <span class="badge">Official Analysis Report</span>
                <div style="font-size:0.75rem; color:#64748b; margin-top:6px;">{date_str}</div>
            </div>
        </div>

        <div class="prediction-hero">
            <div style="font-size: 3rem;">{emoji}</div>
            <div class="emotion-title">{emotion}</div>
            <div style="font-size: 1.15rem; color: #94a3b8; margin-top: 4px;">
                Confidence: <strong style="color:#f8fafc;">{confidence_pct}</strong> &bull;
                Level: <strong style="color:{color};">{confidence_level}</strong>
            </div>
        </div>

        <div class="section-title">📁 Session & Audio Metadata</div>
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">File Name</div>
                <div class="meta-val">{audio_name}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Duration</div>
                <div class="meta-val">{duration_sec:.2f} seconds</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Sample Rate</div>
                <div class="meta-val">{audio_sr} Hz</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Active Model</div>
                <div class="meta-val">{model_name}</div>
            </div>
        </div>

        <div class="section-title">🎯 Top 3 Emotion Probabilities</div>
        <div style="margin-bottom: 20px;">
            {top_3_html}
        </div>

        <div class="section-title">📊 Acoustic Signal Features</div>
        {features_html}

        <div class="section-title">🌊 Acoustic Visualizations</div>
        <div class="chart-box">
            <img src="data:image/png;base64,{waveform_b64}" alt="Waveform">
        </div>
        <div class="chart-box">
            <img src="data:image/png;base64,{spectrogram_b64}" alt="Log-Mel Spectrogram">
        </div>

        <div class="section-title">🧠 Model & Methodology Information</div>
        <div style="background:#131b2e; padding:16px; border-radius:8px; border:1px solid #1e293b; font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
            <strong>Architecture:</strong> Spectrogram-based Deep Convolutional & Temporal Recurrent Neural Network (CNN-LSTM).<br>
            <strong>Feature Space:</strong> 128-band Mel-Filterbank Spectrogram, N_FFT=2048, Hop Length=512, SR=22050Hz.<br>
            <strong>Dataset:</strong> Trained on RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song), with speaker-independent actor partitions to strictly eliminate speaker identity leakage.
        </div>

        <div class="disclaimer">
            <strong>Ethical & Academic Disclaimer:</strong> This voice analysis report is generated by an automated deep learning model for research, educational, and demonstration purposes. Voice-based emotion recognition relies on acoustic feature correlations in simulated dataset recordings. It is not an assessment of clinical mental health or actual human psychological intent.
        </div>

        <div class="footer">
            Generated by EMOTIVA Speech Emotion Intelligence System &bull; All Rights Reserved
        </div>
    </div>
</body>
</html>
"""
    return html
