"""
End-to-End Navigation, Feature & Architecture Verification Tests
Verifies that all 7 navigation destinations, dual models, database persistence,
typography rules, and audio pipelines operate with 100% reliability.
"""
import os
import sys
import unittest
import numpy as np

# Ensure root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from src.preprocessing import load_audio_raw, decode_audio_bytes, audio_to_wav_bytes
from src.feature_extraction import extract_mel_spectrogram, extract_mfcc, extract_audio_features
from src.predict import load_model_and_encoder, predict_emotion, predict_segments
from src.database import (
    init_db, save_prediction, get_predictions, get_prediction_by_id,
    update_prediction, delete_prediction, get_history_stats
)
from src.visualization import _LAYOUT


class TestNavigationAndE2E(unittest.TestCase):
    def setUp(self):
        self.test_db = os.path.join(config.RESULTS_DIR, "test_nav_e2e.db")
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        init_db(self.test_db)

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_01_all_navigation_pages_configured(self):
        """Verify that all 7 navigation modules have icons, titles, and descriptions."""
        expected_pages = ["HOME", "ANALYZE", "AUDIO LAB", "INSIGHTS", "MODELS", "HISTORY", "ABOUT"]
        from app import PAGES, PAGE_CONFIG
        self.assertEqual(PAGES, expected_pages)
        for p in expected_pages:
            self.assertIn(p, PAGE_CONFIG)
            self.assertTrue(len(PAGE_CONFIG[p]["icon"]) > 0)
            self.assertTrue(len(PAGE_CONFIG[p]["title"]) > 0)
            self.assertTrue(len(PAGE_CONFIG[p]["desc"]) > 0)

    def test_02_dual_models_loadable(self):
        """Verify that both CNN-LSTM and CNN models can be loaded without error."""
        m_lstm, enc_lstm = load_model_and_encoder("cnn_lstm")
        self.assertIsNotNone(m_lstm)
        self.assertIsNotNone(enc_lstm)

        m_cnn, enc_cnn = load_model_and_encoder("cnn")
        self.assertIsNotNone(m_cnn)
        self.assertIsNotNone(enc_cnn)

    def test_03_samples_available_and_decodable(self):
        """Verify that demo samples exist and decode cleanly into numpy arrays."""
        for emotion in config.EMOTIONS:
            sample_path = os.path.join(config.PROJECT_ROOT, "samples", f"demo_{emotion}.wav")
            if not os.path.exists(sample_path):
                sample_path = os.path.join(config.PROJECT_ROOT, "samples", f"sample_{emotion}.wav")
            self.assertTrue(os.path.exists(sample_path), f"Sample missing for emotion: {emotion}")
            y, sr = load_audio_raw(sample_path, sr=config.SR)
            self.assertGreater(len(y), 0)
            self.assertEqual(sr, config.SR)

    def test_04_full_predict_and_persist_workflow(self):
        """Verify end-to-end inference and persistence into SQLite database."""
        m_lstm, enc_lstm = load_model_and_encoder("cnn_lstm")
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        y = 0.4 * np.sin(2 * np.pi * 440 * t)

        result = predict_emotion(y, m_lstm, enc_lstm, sr=sr)
        self.assertIn('emotion', result)
        self.assertIn('confidence', result)

        feats = extract_audio_features(y, sr=sr)

        top_3_str = ", ".join([f"{item['emotion']}: {item['confidence_pct']}%" for item in result['top_3']])
        rec_id = save_prediction(
            filename="nav_test_voice.wav",
            duration=duration,
            model="CNN-LSTM",
            emotion=result['emotion'],
            confidence=result['confidence'],
            confidence_pct=result['confidence_pct'],
            confidence_level=result['confidence_level'],
            top_3=top_3_str,
            probabilities=result['probabilities'],
            features=feats,
            user_notes="Audit workflow verified",
            user_tag="Acoustic Benchmark",
            db_path=self.test_db
        )
        self.assertGreater(rec_id, 0)

        # Retrieve and verify persistence
        saved = get_prediction_by_id(rec_id, db_path=self.test_db)
        self.assertIsNotNone(saved)
        self.assertEqual(saved['emotion'], result['emotion'])
        self.assertEqual(saved['user_tag'], "Acoustic Benchmark")

        # Update note
        update_prediction(rec_id, user_notes="Updated note via audit test", is_starred=1, db_path=self.test_db)
        updated = get_prediction_by_id(rec_id, db_path=self.test_db)
        self.assertEqual(updated['user_notes'], "Updated note via audit test")
        self.assertEqual(updated['is_starred'], 1)

    def test_05_strict_typography_enforced(self):
        """Verify that BD Supper and Forum are declared in CSS and visualizations."""
        from app import CSS_THEME
        self.assertIn("BD Supper", CSS_THEME)
        self.assertIn("Forum", CSS_THEME)

        # Plotly layout typography
        self.assertIn("Forum", _LAYOUT['font']['family'])
        self.assertIn("BD Supper", _LAYOUT['title_font']['family'])


if __name__ == '__main__':
    unittest.main()
