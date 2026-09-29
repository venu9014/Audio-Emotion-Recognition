"""
EMOTIVA — AI Speech Emotion Intelligence
Project-wide configuration.
"""
import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'RAVDESS')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Audio parameters
SR = 22050
SAMPLE_RATE = SR  # alias
DURATION = 3  # seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40

# Supported Audio Formats
SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.aac']
AUDIO_MIME_TYPES = {
    '.mp3': 'audio/mp3',
    '.wav': 'audio/wav',
    '.ogg': 'audio/ogg',
    '.flac': 'audio/flac',
    '.m4a': 'audio/mp4',
    '.aac': 'audio/aac',
    '.wma': 'audio/x-ms-wma'
}

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

# Data augmentation flags
ENABLE_AUGMENTATION = True

# Emotion labels mapping for RAVDESS
# RAVDESS filename: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
EMOTION_LABELS = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}

EMOTIONS = list(EMOTION_LABELS.values())

EMOTION_COLORS = {
    'happy': '#f59e0b',
    'sad': '#3b82f6',
    'angry': '#ef4444',
    'fearful': '#8b5cf6',
    'disgust': '#10b981',
    'surprised': '#f97316',
    'calm': '#06b6d4',
    'neutral': '#6b7280'
}

EMOTION_EMOJIS = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😊',
    'sad': '😢',
    'angry': '😡',
    'fearful': '😨',
    'disgust': '🤢',
    'surprised': '😲'
}

EMOTION_DESCRIPTIONS = {
    'happy': 'Associated with increased pitch, higher energy, and faster tempo in speech.',
    'sad': 'Characterized by lower pitch, reduced energy, slower speech rate, and downward intonation.',
    'angry': 'Exhibits high energy, elevated pitch range, fast speech rate, and sharp spectral patterns.',
    'fearful': 'Shows irregular pitch patterns, breathiness, and tension in vocal quality.',
    'disgust': 'Features low pitch, slow articulation, and compressed spectral range.',
    'surprised': 'Marked by sudden pitch rise, wide dynamic range, and short burst patterns.',
    'calm': 'Steady pitch, moderate energy, regular rhythm, and smooth spectral contours.',
    'neutral': 'Baseline speech with moderate pitch, energy, and pace — no strong emotional markers.'
}

# Confidence thresholds
CONFIDENCE_HIGH = 0.7
CONFIDENCE_MODERATE = 0.4
