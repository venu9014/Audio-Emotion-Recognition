"""
EMOTIVA — Audio preprocessing module.
Handles RAVDESS metadata parsing, audio loading, validation, splitting, and augmentation.
"""
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Optional, Dict, Any


def validate_audio_file(
    file_path: str,
    min_duration: float = 0.5,
    max_duration: float = 120.0
) -> Tuple[bool, str, Optional[float], Optional[int]]:
    """
    Validate audio file for format, corruption, emptiness, and duration.

    Returns:
        Tuple of (is_valid, message, duration_seconds, sample_rate)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", None, None

    file_size = os.path.getsize(file_path)
    if file_size <= 44:
        return False, "Audio file is empty or contains no audio data (0 or invalid byte length).", None, None

    try:
        # Check audio info using soundfile
        info = sf.info(file_path)
        duration = float(info.duration)
        sr = int(info.samplerate)

        if duration < min_duration:
            return False, f"Audio is too short ({duration:.2f}s). Minimum required duration is {min_duration}s.", duration, sr

        if duration > max_duration:
            return False, f"Audio is too long ({duration:.1f}s). Maximum allowed duration is {max_duration}s.", duration, sr

        # Test loading a small portion to ensure decodable
        y_test, _ = librosa.load(file_path, sr=None, duration=min(duration, 1.0), mono=True)
        if len(y_test) == 0:
            return False, "Audio file contains zero decodable samples.", duration, sr

        if np.isnan(y_test).any() or np.isinf(y_test).any():
            return False, "Audio contains corrupted numerical data (NaN or Inf values).", duration, sr

        return True, "Audio file is valid.", duration, sr

    except sf.LibsndfileError as e:
        return False, f"Unsupported audio format or corrupted header: {str(e)}", None, None
    except Exception as e:
        return False, f"Unable to read audio file: {str(e)}", None, None


def load_ravdess_metadata(dataset_path: str) -> pd.DataFrame:
    """
    Parse RAVDESS filenames to extract emotion labels, actor IDs, etc.
    RAVDESS filename format: {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
    Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    """
    metadata = []
    if not os.path.exists(dataset_path):
        return pd.DataFrame()

    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if file.endswith('.wav') and not file.startswith('.'):
                file_path = os.path.join(root, file)
                parts = file.split('.')[0].split('-')
                if len(parts) == 7:
                    try:
                        metadata.append({
                            'file_path': file_path,
                            'filename': file,
                            'modality': int(parts[0]),
                            'vocal_channel': int(parts[1]),
                            'emotion': int(parts[2]),
                            'intensity': int(parts[3]),
                            'statement': int(parts[4]),
                            'repetition': int(parts[5]),
                            'actor': int(parts[6])
                        })
                    except ValueError:
                        continue

    df = pd.DataFrame(metadata)
    if len(df) > 0:
        df = df.sort_values('file_path').reset_index(drop=True)
    return df


def load_audio(file_path: str, sr: int = 22050, duration: float = 3.0) -> Tuple[np.ndarray, int]:
    """
    Load audio, convert to mono, resample, normalize amplitude, trim silence, pad/crop to fixed length.

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)

        # Trim silence
        if len(y) > 0:
            y, _ = librosa.effects.trim(y, top_db=20)

        # Normalize amplitude
        if len(y) > 0 and np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y)

        # Pad or crop to target duration
        target_length = int(sr * duration)
        if len(y) > target_length:
            start = (len(y) - target_length) // 2
            y = y[start:start + target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')

        return y, sr
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return np.zeros(int(sr * duration)), sr


def load_audio_raw(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio without trimming/padding — for full-length acoustic visualization.
    """
    try:
        y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)
        if len(y) > 0 and np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y)
        return y, sr
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return np.zeros(sr), sr


def actor_wise_split(
    metadata_df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by actor IDs to prevent speaker leakage.
    Ensures training, validation, and test sets have mutually exclusive speakers.
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
    train_idx, temp_idx = next(gss.split(metadata_df, groups=metadata_df['actor']))

    train_df = metadata_df.iloc[train_idx].copy()
    temp_df = metadata_df.iloc[temp_idx].copy()

    test_ratio = test / (val + test)
    gss_temp = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['actor']))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


def augment_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply random augmentation: noise, pitch shift, time stretch, or time shift.
    Only use for training data.
    """
    if len(y) == 0:
        return y

    y_aug = y.copy()
    aug_type = np.random.choice(['noise', 'pitch', 'stretch', 'shift', 'none'])

    if aug_type == 'noise':
        noise = np.random.normal(0, 0.005, len(y))
        y_aug = y + noise
    elif aug_type == 'pitch':
        n_steps = np.random.randint(-2, 3)
        y_aug = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    elif aug_type == 'stretch':
        rate = np.random.uniform(0.85, 1.15)
        y_aug = librosa.effects.time_stretch(y=y, rate=rate)
        target_length = len(y)
        if len(y_aug) > target_length:
            y_aug = y_aug[:target_length]
        else:
            y_aug = np.pad(y_aug, (0, target_length - len(y_aug)), mode='constant')
    elif aug_type == 'shift':
        shift_amount = np.random.randint(1, max(2, int(sr * 0.3)))
        direction = np.random.choice(['left', 'right'])
        if direction == 'right':
            y_aug = np.roll(y_aug, shift_amount)
            y_aug[:shift_amount] = 0
        else:
            y_aug = np.roll(y_aug, -shift_amount)
            y_aug[-shift_amount:] = 0

    return y_aug
