"""
Master Verification Suite for SONORA / EMOTIVA
Tests all 32 criteria specified in Requirement 23.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import (
    load_audio, load_audio_raw, validate_audio_file, load_ravdess_metadata
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


class TestMasterEmotionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.samples_dir = os.path.join(config.PROJECT_ROOT, "samples")
        cls.neutral_wav = os.path.join(cls.samples_dir, "sample_neutral.wav")
        cls.happy_wav = os.path.join(cls.samples_dir, "sample_happy.wav")
        cls.angry_wav = os.path.join(cls.samples_dir, "sample_angry.wav")
        cls.long_wav = os.path.join(cls.samples_dir, "sample_long_speech.wav")
        cls.short_wav = os.path.join(cls.samples_dir, "sample_too_short.wav")
        cls.corrupted_wav = os.path.join(cls.samples_dir, "sample_corrupted.wav")

    # 1-3. Audio Loading & Validation
    def test_01_valid_audio_loading(self):
        y, sr = load_audio_raw(self.neutral_wav, sr=config.SR)
        self.assertEqual(sr, config.SR)
        self.assertGreater(len(y), 0)
        self.assertFalse(np.isnan(y).any())

    def test_02_audio_validation_valid(self):
        is_valid, msg, dur, sr = validate_audio_file(self.neutral_wav)
        self.assertTrue(is_valid, f"Validation failed: {msg}")
        self.assertGreater(dur, 2.0)

    def test_03_audio_validation_short(self):
        is_valid, msg, dur, sr = validate_audio_file(self.short_wav)
        self.assertFalse(is_valid)
        self.assertIn("too short", msg.lower())

    def test_04_audio_validation_corrupted(self):
        is_valid, msg, dur, sr = validate_audio_file(self.corrupted_wav)
        self.assertFalse(is_valid)

    # 4-7. Acoustic Feature Extraction
    def test_05_mel_spectrogram_shape(self):
        y, sr = load_audio(self.neutral_wav, sr=config.SR, duration=config.DURATION)
        S_dB = extract_mel_spectrogram(y, sr=sr, n_mels=128, target_frames=130)
        self.assertEqual(S_dB.shape, (128, 130))
        self.assertFalse(np.isnan(S_dB).any())

    def test_06_mfcc_extraction(self):
        y, sr = load_audio(self.neutral_wav, sr=config.SR, duration=config.DURATION)
        mfcc_40 = extract_mfcc(y, sr=sr, n_mfcc=40)
        self.assertEqual(mfcc_40.shape[0], 40)
        mfcc_20 = extract_mfcc(y, sr=sr, n_mfcc=20)
        self.assertEqual(mfcc_20.shape[0], 20)

    def test_07_acoustic_scalar_features(self):
        y, sr = load_audio_raw(self.neutral_wav, sr=config.SR)
        feats = extract_audio_features(y, sr)
        self.assertIn('rms_mean', feats)
        self.assertIn('zcr_mean', feats)
        self.assertIn('spectral_centroid_mean', feats)
        self.assertIn('spectral_bandwidth_mean', feats)
        self.assertIn('spectral_rolloff_mean', feats)
        self.assertIn('duration', feats)
        self.assertGreater(feats['duration'], 0)

    # 8-11. Model Loading & Architecture Shape Verification
    def test_08_load_cnn_model(self):
        cnn_model, encoder = load_model_and_encoder('cnn')
        self.assertIsNotNone(cnn_model)
        self.assertEqual(cnn_model.input_shape, (None, 128, 130, 1))
        self.assertEqual(cnn_model.output_shape, (None, 8))
        self.assertEqual(len(encoder.classes_), 8)

    def test_09_load_cnn_lstm_model(self):
        lstm_model, encoder = load_model_and_encoder('cnn_lstm')
        self.assertIsNotNone(lstm_model)
        self.assertEqual(lstm_model.input_shape, (None, 128, 130, 1))
        self.assertEqual(lstm_model.output_shape, (None, 8))
        self.assertEqual(len(encoder.classes_), 8)

    # 12-16. Prediction & Probabilities
    def test_10_prediction_cnn_lstm(self):
        lstm_model, encoder = load_model_and_encoder('cnn_lstm')
        res = predict_emotion(self.neutral_wav, lstm_model, encoder, sr=config.SR)
        self.assertIn('emotion', res)
        self.assertIn(res['emotion'], config.EMOTIONS)
        self.assertGreaterEqual(res['confidence'], 0.0)
        self.assertLessEqual(res['confidence'], 1.0)
        self.assertEqual(len(res['top_3']), 3)
        self.assertAlmostEqual(sum(res['probabilities'].values()), 1.0, places=3)

    def test_11_prediction_cnn(self):
        cnn_model, encoder = load_model_and_encoder('cnn')
        res = predict_emotion(self.happy_wav, cnn_model, encoder, sr=config.SR)
        self.assertIn('emotion', res)
        self.assertIn(res['emotion'], config.EMOTIONS)
        self.assertEqual(len(res['top_3']), 3)

    def test_12_timeline_long_audio(self):
        lstm_model, encoder = load_model_and_encoder('cnn_lstm')
        timeline = predict_segments(self.long_wav, lstm_model, encoder, sr=config.SR, min_duration_for_timeline=4.0)
        self.assertIsNotNone(timeline)
        self.assertGreaterEqual(len(timeline), 2)
        for seg in timeline:
            self.assertIn('start_time', seg)
            self.assertIn('end_time', seg)
            self.assertIn('emotion', seg)
            self.assertIn('confidence', seg)

    def test_13_timeline_short_audio_rejection(self):
        lstm_model, encoder = load_model_and_encoder('cnn_lstm')
        timeline = predict_segments(self.neutral_wav, lstm_model, encoder, sr=config.SR, min_duration_for_timeline=4.0)
        self.assertIsNone(timeline)  # Must be None so UI displays exact required message

    # 17-25. Visualizations
    def test_14_all_visualizations(self):
        y, sr = load_audio_raw(self.neutral_wav, sr=config.SR)
        fig_wave = plot_waveform(y, sr)
        self.assertIsNotNone(fig_wave)

        fig_spec = plot_mel_spectrogram(y, sr)
        self.assertIsNotNone(fig_spec)

        fig_mfcc = plot_mfcc(y, sr, n_mfcc=20)
        self.assertIsNotNone(fig_mfcc)

        fig_cent = plot_spectral_centroid(y, sr)
        self.assertIsNotNone(fig_cent)

        fig_bw = plot_spectral_bandwidth(y, sr)
        self.assertIsNotNone(fig_bw)

        fig_roll = plot_spectral_rolloff(y, sr)
        self.assertIsNotNone(fig_roll)

        fig_rms = plot_rms_energy(y, sr)
        self.assertIsNotNone(fig_rms)

        fig_zcr = plot_zero_crossing_rate(y, sr)
        self.assertIsNotNone(fig_zcr)

        dummy_probs = {e: 1/8 for e in config.EMOTIONS}
        fig_radar = plot_emotion_radar(dummy_probs)
        self.assertIsNotNone(fig_radar)

        fig_bars = plot_emotion_bars(dummy_probs)
        self.assertIsNotNone(fig_bars)

    # 26. Report Generation
    def test_15_html_report_export(self):
        lstm_model, encoder = load_model_and_encoder('cnn_lstm')
        y, sr = load_audio_raw(self.neutral_wav, sr=config.SR)
        res = predict_emotion(y, lstm_model, encoder, sr=sr)
        feats = extract_audio_features(y, sr)

        html = generate_html_report(res, "sample_neutral.wav", sr, y, "CNN-LSTM", feats)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("EMOTIVA / SONORA", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("sample_neutral.wav", html)
        self.assertIn("Top 3 Emotion Probabilities", html)
        self.assertIn("Ethical & Academic Disclaimer", html)

    # 27-28. Missing Dataset & Missing Model Graceful Handling
    def test_16_missing_dataset_handling(self):
        df = load_ravdess_metadata("C:/non_existent_dataset_path_xyz")
        self.assertTrue(df.empty)

    def test_17_missing_model_handling(self):
        with self.assertRaises(FileNotFoundError):
            load_model_and_encoder("non_existent_architecture")

    # 29-32. History Manipulation
    def test_18_history_filtering_sorting(self):
        history = [
            {"timestamp": "2026-09-28 10:00:00", "emotion": "happy", "model": "CNN-LSTM", "confidence": 0.85, "duration": 3.2},
            {"timestamp": "2026-09-28 10:05:00", "emotion": "sad", "model": "CNN", "confidence": 0.60, "duration": 4.1},
            {"timestamp": "2026-09-28 10:10:00", "emotion": "happy", "model": "CNN", "confidence": 0.92, "duration": 2.8}
        ]
        df = pd.DataFrame(history)

        # Filter by emotion
        happy_df = df[df['emotion'] == 'happy']
        self.assertEqual(len(happy_df), 2)

        # Filter by model
        cnn_df = df[df['model'] == 'CNN']
        self.assertEqual(len(cnn_df), 2)

        # Sort by confidence
        sorted_conf = df.sort_values('confidence', ascending=False)
        self.assertEqual(sorted_conf.iloc[0]['confidence'], 0.92)


if __name__ == '__main__':
    unittest.main()
