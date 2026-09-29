"""
EMOTIVA — Prediction module.
Handles model loading, single-file prediction, and segment-level timeline prediction.
"""
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from typing import Union, Tuple, Dict, Any, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import load_audio
from src.feature_extraction import extract_mel_spectrogram


def load_model_and_encoder(model_type: str = 'cnn_lstm') -> Tuple[tf.keras.Model, Any]:
    """
    Load a trained Keras model and the label encoder.

    Args:
        model_type: 'cnn' or 'cnn_lstm'

    Returns:
        Tuple of (model, label_encoder)

    Raises:
        FileNotFoundError: If model or encoder files are missing.
    """
    model_path = os.path.join(config.MODELS_DIR, f"{model_type}_model.keras")
    encoder_path = os.path.join(config.MODELS_DIR, 'label_encoder.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train with: python src/train.py --model {model_type}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}. Train a model first.")

    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    return model, label_encoder


def get_confidence_level(confidence: float) -> str:
    """Return qualitative confidence level based on thresholds."""
    if confidence >= config.CONFIDENCE_HIGH:
        return 'High'
    elif confidence >= config.CONFIDENCE_MODERATE:
        return 'Moderate'
    else:
        return 'Low'


def predict_emotion(
    audio_input: Union[str, np.ndarray],
    model: tf.keras.Model,
    label_encoder: Any,
    sr: int = 22050
) -> Dict[str, Any]:
    """
    Predict emotion from audio file path or numpy array.

    Args:
        audio_input: File path (str) or raw audio signal (np.ndarray)
        model: Trained Keras model
        label_encoder: Fitted sklearn LabelEncoder
        sr: Sample rate

    Returns:
        Dict with keys: emotion, confidence, confidence_level, probabilities, top_3
    """
    import librosa
    if isinstance(audio_input, str):
        y, _ = load_audio(audio_input, sr=sr, duration=config.DURATION)
    else:
        y = audio_input.copy()
        if np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y)
        target_length = int(sr * config.DURATION)
        if len(y) > target_length:
            start = (len(y) - target_length) // 2
            y = y[start:start + target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')

    # Extract mel-spectrogram: shape (128, 130)
    S = extract_mel_spectrogram(
        y, sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )

    # Ensure shape is exactly (1, n_mels, 130, 1)
    if S.shape[1] > 130:
        S = S[:, :130]
    elif S.shape[1] < 130:
        S = np.pad(S, ((0, 0), (0, 130 - S.shape[1])), mode='constant')

    S = np.expand_dims(S, axis=-1)
    S = np.expand_dims(S, axis=0)

    # Predict
    probabilities = model.predict(S, verbose=0)[0]

    # Get top prediction
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])

    # Map class back to emotion name
    raw_label = label_encoder.inverse_transform([pred_idx])[0]
    emotion = config.EMOTION_LABELS.get(raw_label, str(raw_label))

    # Build probability dict for all emotions
    prob_dict = {}
    for i, cls in enumerate(label_encoder.classes_):
        emo_name = config.EMOTION_LABELS.get(cls, f"class_{cls}")
        prob_dict[emo_name] = float(probabilities[i])

    # Top-3 predictions
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_3 = [{'emotion': e, 'confidence': p, 'confidence_pct': round(p * 100, 1)} for e, p in sorted_probs[:3]]

    return {
        'emotion': emotion,
        'confidence': confidence,
        'confidence_pct': round(confidence * 100, 1),
        'confidence_level': get_confidence_level(confidence),
        'probabilities': prob_dict,
        'top_3': top_3,
        'raw_probabilities': probabilities.tolist()
    }


def predict_segments(
    audio_input: Union[str, np.ndarray],
    model: tf.keras.Model,
    label_encoder: Any,
    sr: int = 22050,
    min_duration_for_timeline: float = 4.0,
    window_duration: float = 3.0,
    hop_duration: float = 1.0
) -> Optional[List[Dict[str, Any]]]:
    """
    Run prediction on sliding time windows for a real emotion timeline.
    Only generates timeline if audio duration >= min_duration_for_timeline.

    Args:
        audio_input: File path or raw audio array
        model: Trained Keras model
        label_encoder: Fitted sklearn LabelEncoder
        sr: Sample rate
        min_duration_for_timeline: Minimum length in seconds required for segment timeline
        window_duration: Window length in seconds (matches model input duration = 3.0s)
        hop_duration: Step length in seconds between consecutive segments

    Returns:
        List of segment dicts, or None if audio is too short.
    """
    import librosa
    if isinstance(audio_input, str):
        y, _ = librosa.load(audio_input, sr=sr, mono=True)
    else:
        y = np.array(audio_input, copy=True)

    total_duration = len(y) / sr
    if total_duration < min_duration_for_timeline:
        return None

    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)

    timeline = []
    start_sample = 0

    while start_sample + window_samples <= len(y):
        segment = y[start_sample:start_sample + window_samples]
        start_time = round(start_sample / sr, 2)
        end_time = round((start_sample + window_samples) / sr, 2)

        # Normalize segment
        if np.max(np.abs(segment)) > 0:
            segment = librosa.util.normalize(segment)

        # Extract Mel-spectrogram
        S = extract_mel_spectrogram(
            segment, sr=sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )

        if S.shape[1] > 130:
            S = S[:, :130]
        elif S.shape[1] < 130:
            S = np.pad(S, ((0, 0), (0, 130 - S.shape[1])), mode='constant')

        S_batch = np.expand_dims(np.expand_dims(S, axis=-1), axis=0)
        probabilities = model.predict(S_batch, verbose=0)[0]

        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])

        raw_label = label_encoder.inverse_transform([pred_idx])[0]
        seg_emotion = config.EMOTION_LABELS.get(raw_label, str(raw_label))

        prob_dict = {}
        for j, cls in enumerate(label_encoder.classes_):
            emo_name = config.EMOTION_LABELS.get(cls, f"class_{cls}")
            prob_dict[emo_name] = float(probabilities[j])

        timeline.append({
            'start_time': start_time,
            'end_time': end_time,
            'time_label': f"{start_time:.1f}s - {end_time:.1f}s",
            'emotion': seg_emotion,
            'confidence': confidence,
            'confidence_pct': round(confidence * 100, 1),
            'probabilities': prob_dict
        })

        start_sample += hop_samples

    return timeline if timeline else None
