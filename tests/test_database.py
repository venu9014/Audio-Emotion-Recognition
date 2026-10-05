"""
Unit and integration tests for SQLite Database module (src/database.py).
Verifies CRUD operations, filtering, search, sorting, update, deletion, and settings persistence.
"""
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import (
    init_db, save_prediction, get_predictions, get_prediction_by_id,
    update_prediction, delete_prediction, clear_all_predictions,
    get_history_stats, get_setting, set_setting
)


class TestDatabaseCRUD(unittest.TestCase):

    def setUp(self):
        # Create unique temporary database for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_emotiva.db")
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_read_prediction(self):
        rec_id = save_prediction(
            filename="test_happy.wav",
            duration=3.2,
            model="CNN-LSTM",
            emotion="happy",
            confidence=0.985,
            confidence_pct="98.5%",
            confidence_level="High",
            top_3="happy (98.5%), surprised (1.2%), calm (0.3%)",
            probabilities={"happy": 0.985, "surprised": 0.012, "calm": 0.003},
            features={"rms_mean": 0.052, "zcr_mean": 0.084},
            user_notes="Clear joyful tone",
            user_tag="Interview",
            is_starred=1,
            db_path=self.db_path
        )
        self.assertIsInstance(rec_id, int)
        self.assertGreater(rec_id, 0)

        # Retrieve by ID
        rec = get_prediction_by_id(rec_id, db_path=self.db_path)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["filename"], "test_happy.wav")
        self.assertEqual(rec["emotion"], "happy")
        self.assertEqual(rec["model"], "CNN-LSTM")
        self.assertEqual(rec["confidence"], 0.985)
        self.assertEqual(rec["user_notes"], "Clear joyful tone")
        self.assertEqual(rec["user_tag"], "Interview")
        self.assertEqual(rec["is_starred"], 1)
        self.assertEqual(rec["probabilities"]["happy"], 0.985)

    def test_filtering_and_search(self):
        # Insert 3 records
        save_prediction("audio_1.wav", 3.0, "CNN", "angry", 0.92, "92.0%", "High", "angry", {"angry": 0.92}, db_path=self.db_path)
        save_prediction("audio_2.wav", 2.5, "CNN-LSTM", "sad", 0.88, "88.0%", "High", "sad", {"sad": 0.88}, user_notes="urgent case", db_path=self.db_path)
        save_prediction("audio_3.wav", 4.0, "CNN-LSTM", "happy", 0.99, "99.0%", "High", "happy", {"happy": 0.99}, is_starred=1, db_path=self.db_path)

        # Filter by emotion
        angry_recs = get_predictions(emotion_filter="angry", db_path=self.db_path)
        self.assertEqual(len(angry_recs), 1)
        self.assertEqual(angry_recs[0]["emotion"], "angry")

        # Filter by model
        cnn_recs = get_predictions(model_filter="CNN", db_path=self.db_path)
        self.assertEqual(len(cnn_recs), 1)

        # Starred only
        starred = get_predictions(starred_only=True, db_path=self.db_path)
        self.assertEqual(len(starred), 1)
        self.assertEqual(starred[0]["filename"], "audio_3.wav")

        # Search query matching notes
        search_res = get_predictions(search_query="urgent", db_path=self.db_path)
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["filename"], "audio_2.wav")

    def test_update_prediction(self):
        rec_id = save_prediction(
            "sample.wav", 3.0, "CNN", "neutral", 0.75, "75.0%", "High", "neutral",
            {"neutral": 0.75}, db_path=self.db_path
        )
        updated = update_prediction(
            rec_id,
            user_notes="Updated notes by reviewer",
            user_tag="Verified",
            is_starred=1,
            db_path=self.db_path
        )
        self.assertTrue(updated)

        rec = get_prediction_by_id(rec_id, db_path=self.db_path)
        self.assertEqual(rec["user_notes"], "Updated notes by reviewer")
        self.assertEqual(rec["user_tag"], "Verified")
        self.assertEqual(rec["is_starred"], 1)

    def test_delete_prediction(self):
        rec_id = save_prediction(
            "temp.wav", 2.0, "CNN", "calm", 0.80, "80.0%", "High", "calm",
            {"calm": 0.80}, db_path=self.db_path
        )
        deleted = delete_prediction(rec_id, db_path=self.db_path)
        self.assertTrue(deleted)

        rec = get_prediction_by_id(rec_id, db_path=self.db_path)
        self.assertIsNone(rec)

    def test_clear_all_and_stats(self):
        save_prediction("1.wav", 3.0, "CNN", "calm", 0.80, "80.0%", "High", "calm", {"calm": 0.8}, db_path=self.db_path)
        save_prediction("2.wav", 3.0, "CNN-LSTM", "fearful", 0.65, "65.0%", "Moderate", "fearful", {"fearful": 0.65}, db_path=self.db_path)

        stats = get_history_stats(db_path=self.db_path)
        self.assertEqual(stats["total_records"], 2)
        self.assertGreater(stats["average_confidence"], 0.70)

        clear_all_predictions(db_path=self.db_path)
        stats_empty = get_history_stats(db_path=self.db_path)
        self.assertEqual(stats_empty["total_records"], 0)

    def test_settings_storage(self):
        set_setting("theme", "editorial-dark", db_path=self.db_path)
        val = get_setting("theme", db_path=self.db_path)
        self.assertEqual(val, "editorial-dark")

        # Update setting
        set_setting("theme", "cyber-cyan", db_path=self.db_path)
        val2 = get_setting("theme", db_path=self.db_path)
        self.assertEqual(val2, "cyber-cyan")


if __name__ == '__main__':
    unittest.main()
