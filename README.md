# 🎙️ SONORA / EMOTIVA — AI Speech Emotion Intelligence

> *"Listen beyond words."*

**Spectrogram-Based CNN-LSTM Deep Speech Emotion Recognition System**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Librosa](https://img.shields.io/badge/Librosa-0.10+-orange)](https://librosa.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)

---

## 📋 System Overview

**SONORA / EMOTIVA** is an end-to-end Speech Emotion Recognition (SER) system engineered to detect and classify human vocal affect directly from raw acoustic waveforms. By transforming speech into 128-band Mel-Filterbank energy representations and streaming them through hybrid **Convolutional Neural Networks and Long Short-Term Memory (CNN-LSTM)** spatiotemporal networks, the system captures both localized spectral formant geometries and long-range temporal prosodic contours.

The project features a high-performance **AI Audio Laboratory** Streamlit interface with dark editorial aesthetic, real-time acoustic signal inspection, functional model switching, segment-level emotion tracking, session audit logs, and self-contained HTML report exports.

---

## ✨ Feature Highlights

### 🔬 Core Affect Recognition
- **8 Discrete Affect Categories**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
- **Dual Inference Architecture with Dynamic Selection**:
  - **CNN-LSTM (Main)**: Spatiotemporal feature extraction via Conv2D + Temporal Permute + LSTM(128).
  - **CNN Baseline**: Pure spatial spectral convolutions + Global Average Pooling.
- **Microphone Recording & Audio Upload**: Supports browser recording via `st.audio_input` and `.wav` file uploads.
- **Pre-Synthesized Demo Audio Library**: Immediate one-click testing of Neutral, Happy, Angry, and 6.5s Multi-Emotion speech without needing external audio.

### 🛡️ Robust Audio Validation (Requirement 14)
- **Empty / 0-byte File Detection**: Rejects empty or corrupt audio with actionable error notices.
- **Duration Boundary Enforcement**: Validates length; handles short audio (<0.5s) and limits long audio safely.
- **Format Integrity**: Strict WAV header validation and non-finite value checking (NaN / Inf protection).

### 🧬 Signal Processing & Audio Lab (Requirement 4 & 20)
- **Time-Domain Waveform**: Normalized acoustic pressure tracking.
- **Log-Mel Spectrogram**: 128 Mel bands with interactive controls for N_MELS, N_FFT, and Hop Length.
- **MFCC Dynamic Inspection**: Interactive coefficient count slider (10 to 40 coefficients) updating in real time.
- **High-Order Spectral Analytics**: Real-time Spectral Centroid ("brightness"), Spectral Bandwidth, and Spectral Rolloff.
- **Energy & Zero Crossing Dynamics**: Vocal loudness (RMS) and sign-polarity frequency changes (ZCR).

### 📊 Results & Visualizations (Requirement 7 & 8)
- **Emotion Constellation**: Radar polygon chart displaying probabilities across all 8 classes.
- **Confidence Ranking**: Primary emotion badge, percentage confidence, and qualitative confidence rating (High, Moderate, Low).
- **Top 3 Progress Bars**: Visual candidate comparison with percentage distributions.
- **Emotion Trajectory Timeline**: Real sliding-window segment-level predictions across time (for audio >= 4.0s).
- **Graceful Short Audio Handling**: Displays *"Audio is too short for segment-level emotion analysis."* when audio is below 4.0s.

### 📜 Session History, Reset & Reports (Requirements 9, 10, 11)
- **Dynamic History Log**: Tracks timestamp, filename, duration, model, predicted emotion, confidence, and top-3 candidates.
- **Interactive History Filtering & Sorting**: Filter by emotion or model; sort by date, confidence, or duration.
- **CSV History Export**: Downloads current filtered session history table.
- **Full Reset Action**: Clears current audio, waveform, features, predictions, and timeline, returning to initial state.
- **Self-Contained HTML Report Export**: Generates printable HTML reports containing audio metadata, acoustic metrics, prediction breakdowns, ethical disclaimers, and **embedded base64 images of both the waveform and Mel-spectrogram**.

---

## 🛠️ Technology Stack

| Category | Component | Purpose |
|---|---|---|
| **Deep Learning** | TensorFlow 2.x / Keras | CNN & CNN-LSTM model definition, training, checkpointing |
| **Audio Processing** | Librosa, SoundFile | Resampling, silence trimming, spectrogram & MFCC extraction |
| **Data Manipulation** | NumPy, Pandas, Scikit-learn | Array math, speaker-independent splits, LabelEncoder |
| **Visualization** | Plotly, Matplotlib | Interactive web plots, dark-themed base64 report rendering |
| **Web Interface** | Streamlit | Responsive, real-time reactive user interface |

---

## 📁 Repository Structure

```
EMOTIVA/
│
├── app.py                      # Main Streamlit application
├── config.py                   # Global audio, training, and emotion configurations
├── requirements.txt            # Python environment dependencies
├── README.md                   # Complete architectural and usage guide
│
├── dataset/
│   └── RAVDESS/                # Official audio directory
│       ├── Actor_01/           # Audio files: 03-01-XX-XX-XX-XX-01.wav
│       ├── Actor_02/
│       └── ...Actor_24/
│
├── models/
│   ├── cnn_model.keras         # Trained CNN baseline model
│   ├── cnn_lstm_model.keras    # Trained CNN-LSTM spatiotemporal model
│   └── label_encoder.pkl       # Fitted 8-class LabelEncoder
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Loading, audio validation, silence trimming, actor-wise split
│   ├── feature_extraction.py   # Mel-spectrogram, MFCC, and scalar acoustic features
│   ├── models.py               # CNN and CNN-LSTM Keras architecture builders
│   ├── train.py                # Model training pipeline with early stopping & checkpoints
│   ├── evaluate.py             # Evaluation on test actors, confusion matrix, classification report
│   ├── predict.py              # Single-file prediction and sliding-window segment timeline
│   ├── visualization.py        # Plotly dark-theme visualizers
│   ├── report.py               # Self-contained HTML report generator with embedded base64 charts
│   ├── download_dataset.py     # Zenodo downloader utility
│   └── create_sample_dataset.py# Synthetic test audio and dataset generator
│
├── samples/                    # Pre-synthesized test audio samples
│   ├── sample_neutral.wav
│   ├── sample_happy.wav
│   ├── sample_angry.wav
│   ├── sample_long_speech.wav  # 6.5s sample for timeline testing
│   ├── sample_too_short.wav    # 0.3s sample for short validation testing
│   └── sample_corrupted.wav    # Corrupted header for format validation testing
│
├── results/
│   ├── cnn_history.json        # CNN training & validation loss/accuracy curves
│   ├── cnn_lstm_history.json   # CNN-LSTM training & validation curves
│   ├── model_comparison.csv    # Real test set comparison table
│   └── evaluation_results.json # Full confusion matrix & classification report JSON
│
└── tests/
    └── test_master.py          # 18-step master test suite covering all 32 requirements
```

---

## 🏗️ Neural Network Architectures

### 1. CNN-LSTM (Primary Spatiotemporal Architecture)
```
Input: (128 Mel Frequency Bands, 130 Time Frames, 1 Channel)
 │
 ├── Conv2D(32, (3, 3), padding='same') + BatchNorm + ReLU
 ├── MaxPooling2D(pool_size=(2, 2)) + Dropout(0.3)
 │
 ├── Conv2D(64, (3, 3), padding='same') + BatchNorm + ReLU
 ├── MaxPooling2D(pool_size=(2, 4)) + Dropout(0.3)
 │
 ├── Permute((2, 1, 3))       # Transpose to (batch, time=16, freq=32, channels=64)
 ├── Reshape((16, 2048))      # Sequence of 16 timesteps, each with 2048 features
 │
 ├── LSTM(128 units, return_sequences=False) + Dropout(0.4)
 ├── Dense(64, activation='relu')
 └── Dense(8, activation='softmax') -> Emotion Probability Distribution
```

### 2. CNN Baseline (Spatial Convolutional Architecture)
```
Input: (128 Mel Frequency Bands, 130 Time Frames, 1 Channel)
 │
 ├── Conv2D(32, (3, 3), padding='same') + BatchNorm + ReLU
 ├── MaxPooling2D(pool_size=(2, 2)) + Dropout(0.3)
 │
 ├── Conv2D(64, (3, 3), padding='same') + BatchNorm + ReLU
 ├── MaxPooling2D(pool_size=(2, 2)) + Dropout(0.3)
 │
 ├── GlobalAveragePooling2D()  # Flattens spatial map into 64-d vector
 ├── Dense(64, activation='relu') + Dropout(0.3)
 └── Dense(8, activation='softmax') -> Emotion Probability Distribution
```

---

## ⚡ Setup & Installation

### 1. Environment Preparation
```bash
# Clone repository
git clone https://github.com/yourusername/emotiva.git
cd emotiva

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate
# Activate environment (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Setup & Training

### 1. Option A: Official RAVDESS Dataset
1. Download `Audio_Speech_Actors_01-24.zip` from [Zenodo (Record 1188976)](https://zenodo.org/records/1188976).
2. Extract into `dataset/RAVDESS/` such that:
   ```
   dataset/RAVDESS/Actor_01/*.wav
   dataset/RAVDESS/Actor_02/*.wav
   ...
   dataset/RAVDESS/Actor_24/*.wav
   ```

### 2. Option B: Synthetic Test Dataset Generator
For immediate offline verification, generate a 4-actor dataset and test audio suite:
```bash
python src/create_sample_dataset.py
```

### 3. Model Training
```bash
# Train CNN Baseline
python src/train.py --model cnn --epochs 30

# Train CNN-LSTM (Recommended)
python src/train.py --model cnn_lstm --epochs 40

# Train with Data Augmentation
python src/train.py --model cnn_lstm --augment
```

### 4. Benchmark Evaluation
Evaluate both models on the held-out test actors (strictly no data leakage):
```bash
python src/evaluate.py
```

---

## 🧪 Comprehensive Verification Suite

Run the full automated test suite verifying all 32 project requirements:
```bash
python -m unittest tests/test_master.py
```
*Outputs:*
```
Ran 18 tests in 27.180s
OK
```

---

## 🚀 Running the Web Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚠️ Academic & Ethical Disclaimers

1. **Non-Clinical Purpose**: This software is intended strictly for academic demonstration, algorithmic benchmarking, and educational research. Model predictions should not be used as clinical diagnostic indicators of mental health or psychological intent.
2. **Supervised Corpus Bias**: Predictions correlate acoustic features with the styles of actors performing in the RAVDESS corpus (North American English). Acoustic variation across languages, accents, and conversational contexts may affect real-world generalization.
3. **Local Privacy**: Audio is processed entirely on the host system and is never uploaded to any remote service.

---

<p align="center">
  <b>SONORA / EMOTIVA</b> &bull; Deep Speech Emotion Intelligence Laboratory
</p>
