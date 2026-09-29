"""
EMOTIVA — Feature extraction module.
Handles Mel-spectrogram, MFCC, and scalar audio feature extraction.
"""
import os
import sys
import numpy as np
import pandas as pd
import librosa
from typing import Dict, Any, Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import load_audio, augment_audio


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    target_frames: int = 130
) -> np.ndarray:
    """
    Extract log-mel spectrogram from audio signal with consistent frame dimension.
    """
    if len(y) == 0:
        return np.zeros((n_mels, target_frames), dtype=np.float32)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    ref_val = np.max(S) if np.max(S) > 0 else 1.0
    S_dB = librosa.power_to_db(S, ref=ref_val)

    if target_frames is not None:
        if S_dB.shape[1] > target_frames:
            S_dB = S_dB[:, :target_frames]
        elif S_dB.shape[1] < target_frames:
            pad_width = target_frames - S_dB.shape[1]
            S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')

    # Map dB from [-80.0, 0.0] to [0.0, 1.0] for stable neural network convergence
    S_norm = (S_dB + 80.0) / 80.0
    S_norm = np.clip(S_norm, 0.0, 1.0)

    return S_norm.astype(np.float32)


def extract_mfcc(y: np.ndarray, sr: int = 22050, n_mfcc: int = 40, hop_length: int = 512) -> np.ndarray:
    """Extract MFCCs from audio signal."""
    if len(y) == 0:
        return np.zeros((n_mfcc, 10), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.astype(np.float32)


def extract_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract scalar audio features for display.
    Returns dict with RMS, ZCR, spectral centroid, bandwidth, rolloff, duration, mean MFCC.
    """
    if len(y) == 0:
        return {
            'rms_mean': 0.0, 'rms_std': 0.0, 'zcr_mean': 0.0,
            'spectral_centroid_mean': 0.0, 'spectral_bandwidth_mean': 0.0,
            'spectral_rolloff_mean': 0.0, 'duration': 0.0,
            'sample_rate': sr, 'mfcc_mean': 0.0
        }

    features = {}

    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = float(np.mean(zcr))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(cent))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(rolloff))

    features['duration'] = float(len(y) / sr)
    features['sample_rate'] = int(sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = float(np.mean(mfcc))

    return features


def prepare_dataset(
    metadata_df: pd.DataFrame,
    dataset_path: str,
    config_module: Any,
    augment: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all audio, extract spectrograms, return X, y arrays ready for model input.
    X shape: (n_samples, n_mels, 130, 1)
    """
    X = []
    y_labels = []
    total = len(metadata_df)

    for idx, (_, row) in enumerate(metadata_df.iterrows()):
        file_path = row['file_path']
        emotion = row['emotion']

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing {idx + 1}/{total}...")

        y, sr = load_audio(file_path, sr=config_module.SR, duration=config_module.DURATION)

        features = extract_mel_spectrogram(
            y, sr=config_module.SR,
            n_mels=config_module.N_MELS,
            n_fft=config_module.N_FFT,
            hop_length=config_module.HOP_LENGTH,
            target_frames=130
        )

        features = np.expand_dims(features, axis=-1)
        X.append(features)
        y_labels.append(emotion)

        if augment and config_module.ENABLE_AUGMENTATION:
            y_aug = augment_audio(y, sr=config_module.SR)
            features_aug = extract_mel_spectrogram(
                y_aug, sr=config_module.SR,
                n_mels=config_module.N_MELS,
                n_fft=config_module.N_FFT,
                hop_length=config_module.HOP_LENGTH,
                target_frames=130
            )
            features_aug = np.expand_dims(features_aug, axis=-1)
            X.append(features_aug)
            y_labels.append(emotion)

    return np.array(X, dtype=np.float32), np.array(y_labels, dtype=np.int32)
