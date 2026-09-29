import io
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import config
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Optional, Dict, Any, Union


def decode_audio_bytes(audio_bytes: bytes, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Decodes audio bytes from any format (WAV, MP3, WebM, Opus, OGG, FLAC, M4A, AAC)
    into a mono float32 numpy array.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        return np.zeros(0, dtype=np.float32), target_sr

    bio = io.BytesIO(audio_bytes)

    # 1. Try PyAV first (universal container decoder for WebM, Opus, MP4, etc.)
    try:
        import av
        container = av.open(bio)
        if len(container.streams.audio) > 0:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
            frames = []
            for packet in container.demux(stream):
                for frame in packet.decode():
                    for res in resampler.resample(frame):
                        frames.append(res.to_ndarray())
            container.close()
            if frames:
                y = np.concatenate(frames, axis=1).squeeze()
                if y.ndim > 1:
                    y = np.mean(y, axis=0)
                return y.astype(np.float32), target_sr
    except Exception:
        pass

    # 2. Try soundfile
    try:
        bio.seek(0)
        y, orig_sr = sf.read(bio)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if orig_sr != target_sr:
            y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        return y.astype(np.float32), target_sr
    except Exception:
        pass

    # 3. Try librosa
    try:
        bio.seek(0)
        y, sr = librosa.load(bio, sr=target_sr, mono=True)
        return y.astype(np.float32), target_sr
    except Exception:
        pass

    return np.zeros(0, dtype=np.float32), target_sr


def audio_to_wav_bytes(y: np.ndarray, sr: int = 22050) -> bytes:
    """Converts a float32 audio numpy array into standard 16-bit PCM WAV bytes."""
    bio = io.BytesIO()
    max_val = np.max(np.abs(y)) if len(y) > 0 else 0
    if max_val > 1.0:
        y_norm = y / max_val
    else:
        y_norm = y
    sf.write(bio, y_norm.astype(np.float32), sr, format='WAV', subtype='PCM_16')
    return bio.getvalue()


def validate_audio_file(
    file_path: str,
    min_duration: float = 0.5,
    max_duration: float = 120.0
) -> Tuple[bool, str, Optional[float], Optional[int]]:
    """
    Validate audio file for format (WAV, MP3, WebM, FLAC, OGG, M4A, etc.), corruption, emptiness, and duration.

    Returns:
        Tuple of (is_valid, message, duration_seconds, sample_rate)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", None, None

    file_size = os.path.getsize(file_path)
    if file_size <= 44:
        return False, "Audio file is empty or contains no audio data (0 or invalid byte length).", None, None

    try:
        duration = None
        sr = None

        # 1. Attempt header inspection via soundfile
        try:
            info = sf.info(file_path)
            duration = float(info.duration)
            sr = int(info.samplerate)
        except Exception:
            pass

        # 2. Attempt PyAV inspection (great for WebM/Opus microphone streams)
        if duration is None or sr is None or duration <= 0:
            try:
                import av
                container = av.open(file_path)
                if len(container.streams.audio) > 0:
                    stream = container.streams.audio[0]
                    sr = stream.rate or 22050
                    if stream.duration is not None and stream.time_base is not None:
                        duration = float(stream.duration * stream.time_base)
                    elif container.duration is not None:
                        duration = float(container.duration / av.time.AV_TIME_BASE)
                container.close()
            except Exception:
                pass

        # 3. Fallback for MP3 or non-standard container formats via librosa
        if duration is None or sr is None or duration <= 0:
            try:
                duration = float(librosa.get_duration(path=file_path))
                y_probe, sr_probe = librosa.load(file_path, sr=None, duration=0.25, mono=True)
                sr = int(sr_probe)
            except Exception:
                # 4. Try decoding full bytes directly
                try:
                    with open(file_path, 'rb') as f:
                        raw_bytes = f.read()
                    y_dec, sr_dec = decode_audio_bytes(raw_bytes, target_sr=config.SR)
                    if len(y_dec) > 0:
                        duration = float(len(y_dec) / sr_dec)
                        sr = int(sr_dec)
                except Exception as dec_err:
                    return False, f"Unsupported audio format or unreadable stream: {str(dec_err)}", None, None

        if duration is None or sr is None or duration <= 0:
            return False, "Unable to determine audio duration or sample rate.", None, None

        if duration < min_duration:
            return False, f"Audio is too short ({duration:.2f}s). Minimum required duration is {min_duration}s.", duration, sr

        if duration > max_duration:
            return False, f"Audio is too long ({duration:.1f}s). Maximum allowed duration is {max_duration}s.", duration, sr

        # Test loading a small portion to ensure decodable audio stream
        y_test = None
        try:
            y_test, _ = librosa.load(file_path, sr=None, duration=min(duration, 1.0), mono=True)
        except Exception:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            y_test, _ = decode_audio_bytes(raw_bytes, target_sr=config.SR)

        if y_test is None or len(y_test) == 0:
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
    RAVDESS filename format: {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.{ext}
    Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    """
    metadata = []
    if not os.path.exists(dataset_path):
        return pd.DataFrame()

    supported_exts = tuple(getattr(config, 'SUPPORTED_AUDIO_EXTENSIONS', ['.wav', '.mp3', '.flac', '.ogg']))

    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if any(file.lower().endswith(ext) for ext in supported_exts) and not file.startswith('.'):
                file_path = os.path.join(root, file)
                parts = os.path.splitext(file)[0].split('-')
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
        try:
            y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)
        except Exception:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            y, loaded_sr = decode_audio_bytes(raw_bytes, target_sr=sr)

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
        try:
            y, loaded_sr = librosa.load(file_path, sr=sr, mono=True)
        except Exception:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            y, loaded_sr = decode_audio_bytes(raw_bytes, target_sr=sr)

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
