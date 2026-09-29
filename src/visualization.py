"""
EMOTIVA — Visualization module.
All functions return Plotly figure objects with dark theme.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# -------------------------------------------------------------------
# Shared layout defaults
# -------------------------------------------------------------------
_LAYOUT = dict(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=40, r=20, t=50, b=40),
    font=dict(family="Inter, sans-serif", size=12, color="#e2e8f0"),
)


def _apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    merged = {**_LAYOUT, **overrides}
    fig.update_layout(**merged)
    return fig


# -------------------------------------------------------------------
# Audio signal visualizations
# -------------------------------------------------------------------

def plot_waveform(y: np.ndarray, sr: int) -> go.Figure:
    """Interactive waveform plot."""
    t = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure(go.Scatter(
        x=t, y=y,
        mode='lines',
        line=dict(color='#06b6d4', width=1),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>'
    ))
    return _apply_layout(fig, title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")


def plot_mel_spectrogram(y: np.ndarray, sr: int,
                         n_mels: int = 128, n_fft: int = 2048,
                         hop_length: int = 512) -> go.Figure:
    """Interactive mel-spectrogram heatmap."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    n_frames = S_dB.shape[1]
    time_axis = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    freq_axis = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)

    fig = go.Figure(go.Heatmap(
        z=S_dB,
        x=time_axis,
        y=freq_axis,
        colorscale='Viridis',
        colorbar=dict(title='dB'),
        hovertemplate='Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Power: %{z:.1f} dB<extra></extra>'
    ))
    return _apply_layout(fig, title="Mel-Spectrogram (Audio DNA)",
                         xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")


def plot_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 40,
              hop_length: int = 512) -> go.Figure:
    """MFCC heatmap."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfccs.shape[1]
    time_axis = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    fig = go.Figure(go.Heatmap(
        z=mfccs,
        x=time_axis,
        y=list(range(1, n_mfcc + 1)),
        colorscale='Magma',
        colorbar=dict(title='Coefficient'),
        hovertemplate='Time: %{x:.2f}s<br>MFCC #%{y}<br>Value: %{z:.2f}<extra></extra>'
    ))
    return _apply_layout(fig, title="MFCC", xaxis_title="Time (s)", yaxis_title="Coefficient #")


def plot_spectral_centroid(y: np.ndarray, sr: int) -> go.Figure:
    """Spectral centroid over time."""
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = np.arange(len(cent))
    t = librosa.frames_to_time(frames, sr=sr)

    fig = go.Figure(go.Scatter(
        x=t, y=cent,
        mode='lines',
        line=dict(color='#f59e0b', width=2),
        fill='tozeroy',
        fillcolor='rgba(245,158,11,0.1)',
        hovertemplate='Time: %{x:.2f}s<br>Centroid: %{y:.0f} Hz<extra></extra>'
    ))
    return _apply_layout(fig, title="Spectral Centroid",
                         xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")


def plot_spectral_bandwidth(y: np.ndarray, sr: int) -> go.Figure:
    """Spectral bandwidth over time."""
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    frames = np.arange(len(bandwidth))
    t = librosa.frames_to_time(frames, sr=sr)

    fig = go.Figure(go.Scatter(
        x=t, y=bandwidth,
        mode='lines',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16,185,129,0.1)',
        hovertemplate='Time: %{x:.2f}s<br>Bandwidth: %{y:.0f} Hz<extra></extra>'
    ))
    return _apply_layout(fig, title="Spectral Bandwidth",
                         xaxis_title="Time (s)", yaxis_title="Bandwidth (Hz)")


def plot_spectral_rolloff(y: np.ndarray, sr: int) -> go.Figure:
    """Spectral rolloff over time."""
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    frames = np.arange(len(rolloff))
    t = librosa.frames_to_time(frames, sr=sr)

    fig = go.Figure(go.Scatter(
        x=t, y=rolloff,
        mode='lines',
        line=dict(color='#8b5cf6', width=2),
        fill='tozeroy',
        fillcolor='rgba(139,92,246,0.1)',
        hovertemplate='Time: %{x:.2f}s<br>Rolloff: %{y:.0f} Hz<extra></extra>'
    ))
    return _apply_layout(fig, title="Spectral Rolloff",
                         xaxis_title="Time (s)", yaxis_title="Rolloff (Hz)")


def plot_rms_energy(y: np.ndarray, sr: int) -> go.Figure:
    """RMS energy over time."""
    rms = librosa.feature.rms(y=y)[0]
    frames = np.arange(len(rms))
    t = librosa.frames_to_time(frames, sr=sr)

    fig = go.Figure(go.Scatter(
        x=t, y=rms,
        mode='lines',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.15)',
        hovertemplate='Time: %{x:.2f}s<br>Energy: %{y:.4f}<extra></extra>'
    ))
    return _apply_layout(fig, title="Voice Energy (RMS)",
                         xaxis_title="Time (s)", yaxis_title="Energy")


def plot_zero_crossing_rate(y: np.ndarray, sr: int) -> go.Figure:
    """Zero crossing rate over time."""
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    frames = np.arange(len(zcr))
    t = librosa.frames_to_time(frames, sr=sr)

    fig = go.Figure(go.Scatter(
        x=t, y=zcr,
        mode='lines',
        line=dict(color='#ef4444', width=2),
        hovertemplate='Time: %{x:.2f}s<br>ZCR: %{y:.4f}<extra></extra>'
    ))
    return _apply_layout(fig, title="Zero Crossing Rate",
                         xaxis_title="Time (s)", yaxis_title="Rate")


# -------------------------------------------------------------------
# Emotion result visualizations
# -------------------------------------------------------------------

def plot_emotion_radar(prob_dict: dict) -> go.Figure:
    """Radar chart showing all emotion probabilities."""
    emotions = list(prob_dict.keys())
    values = [prob_dict[e] for e in emotions]

    # Close the polygon
    emotions_closed = emotions + [emotions[0]]
    values_closed = values + [values[0]]

    colors = [config.EMOTION_COLORS.get(e, '#6b7280') for e in emotions]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=[e.capitalize() for e in emotions_closed],
        fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6, color='#3b82f6'),
        hovertemplate='%{theta}: %{r:.1%}<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=True,
                            tickformat='.0%', gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        **_LAYOUT,
        title="Emotion Constellation"
    )
    return fig


def plot_emotion_bars(prob_dict: dict) -> go.Figure:
    """Horizontal bar chart of emotion probabilities."""
    sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    emotions = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [config.EMOTION_COLORS.get(e, '#6b7280') for e in emotions]

    fig = go.Figure(go.Bar(
        x=values,
        y=[e.capitalize() for e in emotions],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition='outside',
        hovertemplate='%{y}: %{x:.1%}<extra></extra>'
    ))
    return _apply_layout(fig, title="Emotion Probabilities",
                         xaxis_title="Probability", xaxis=dict(range=[0, 1.05]),
                         yaxis=dict(categoryorder='total ascending'),
                         height=350)


def plot_emotion_timeline(timeline: list) -> go.Figure:
    """
    Timeline showing dominant emotion per segment and probability curves.
    timeline: list of dicts from predict_segments()
    """
    fig = go.Figure()

    if not timeline:
        return _apply_layout(fig, title="Emotion Timeline")

    all_emotions = list(timeline[0]['probabilities'].keys())
    times = [seg['start_time'] for seg in timeline]

    for emo in all_emotions:
        emo_probs = [seg['probabilities'].get(emo, 0) for seg in timeline]
        color = config.EMOTION_COLORS.get(emo, '#6b7280')
        fig.add_trace(go.Scatter(
            x=times, y=emo_probs,
            mode='lines+markers',
            name=emo.capitalize(),
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f'{emo.capitalize()}<br>Time: %{{x}}s<br>Prob: %{{y:.1%}}<extra></extra>'
        ))

    # Add dominant emotion annotations
    for seg in timeline:
        fig.add_annotation(
            x=seg['start_time'],
            y=seg['confidence'],
            text=config.EMOTION_EMOJIS.get(seg['emotion'], '🎭'),
            showarrow=False,
            font=dict(size=16),
            yshift=15
        )

    return _apply_layout(fig, title="Emotion Timeline (Segment-Level Estimated Emotion)",
                         xaxis_title="Time (s)", yaxis_title="Probability",
                         yaxis=dict(range=[0, 1.1]),
                         legend=dict(orientation='h', y=-0.15),
                         height=400)


# -------------------------------------------------------------------
# Model evaluation visualizations
# -------------------------------------------------------------------

def plot_confusion_matrix(cm, labels: list) -> go.Figure:
    """Interactive confusion matrix heatmap."""
    cm_array = np.array(cm)
    display_labels = [l.capitalize() if isinstance(l, str) else config.EMOTION_LABELS.get(l, str(l)).capitalize()
                      for l in labels]

    fig = go.Figure(go.Heatmap(
        z=cm_array,
        x=display_labels,
        y=display_labels,
        colorscale='Blues',
        showscale=True,
        text=cm_array,
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    return _apply_layout(fig, title="Confusion Matrix",
                         xaxis_title="Predicted Emotion",
                         yaxis_title="True Emotion",
                         yaxis=dict(autorange='reversed'),
                         height=500)


def plot_training_curves(history: dict) -> go.Figure:
    """Accuracy and Loss training curves side by side."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))

    epochs = list(range(1, len(history.get('accuracy', [])) + 1))

    if 'accuracy' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['accuracy'], name='Train Accuracy',
                       line=dict(color='#06b6d4', width=2)),
            row=1, col=1
        )
    if 'val_accuracy' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Accuracy',
                       line=dict(color='#f59e0b', width=2, dash='dash')),
            row=1, col=1
        )
    if 'loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['loss'], name='Train Loss',
                       line=dict(color='#3b82f6', width=2)),
            row=1, col=2
        )
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                       line=dict(color='#ef4444', width=2, dash='dash')),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    return _apply_layout(fig, title="Training History", height=400)


def plot_emotion_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of emotion distribution in dataset."""
    counts = df['emotion'].value_counts().sort_index()
    labels = [config.EMOTION_LABELS.get(k, str(k)).capitalize() for k in counts.index]
    colors = [config.EMOTION_COLORS.get(config.EMOTION_LABELS.get(k, ''), '#6b7280') for k in counts.index]

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition='outside',
        hovertemplate='%{x}: %{y} samples<extra></extra>'
    ))
    return _apply_layout(fig, title="Dataset Emotion Distribution",
                         xaxis_title="Emotion", yaxis_title="Number of Samples")


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing CNN vs CNN-LSTM metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    fig = go.Figure()
    colors = ['#3b82f6', '#f59e0b']

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        vals = [row.get(m, 0) for m in available_metrics]
        fig.add_trace(go.Bar(
            name=row.get('Model', f'Model {i+1}'),
            x=available_metrics,
            y=vals,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.3f}" for v in vals],
            textposition='outside'
        ))

    fig.update_layout(barmode='group')
    return _apply_layout(fig, title="CNN vs CNN-LSTM Performance Comparison",
                         xaxis_title="Metric", yaxis_title="Score",
                         yaxis=dict(range=[0, 1.1]), height=400)
