"""
EMOTIVA / SONORA — SQLite Database Module
Handles persistent storage and CRUD operations for predictions, user notes, tags, and settings.
"""
import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

import config

DB_PATH = os.path.join(config.RESULTS_DIR, "emotiva_history.db")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return an active connection to the SQLite database."""
    target_path = db_path or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables if they do not already exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        filename TEXT NOT NULL,
        duration REAL NOT NULL,
        model TEXT NOT NULL,
        emotion TEXT NOT NULL,
        confidence REAL NOT NULL,
        confidence_pct TEXT NOT NULL,
        confidence_level TEXT NOT NULL,
        top_3 TEXT NOT NULL,
        probabilities_json TEXT NOT NULL,
        features_json TEXT,
        user_notes TEXT DEFAULT '',
        user_tag TEXT DEFAULT 'General',
        is_starred INTEGER DEFAULT 0
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    filename: str,
    duration: float,
    model: str,
    emotion: str,
    confidence: float,
    confidence_pct: str,
    confidence_level: str,
    top_3: str,
    probabilities: Dict[str, float],
    features: Optional[Dict[str, Any]] = None,
    user_notes: str = "",
    user_tag: str = "General",
    is_starred: int = 0,
    db_path: Optional[str] = None
) -> int:
    """
    Insert a new prediction record into the database (CREATE).
    Returns the newly created record ID.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prob_json = json.dumps(probabilities)
    feat_json = json.dumps(features) if features else "{}"

    cursor.execute("""
    INSERT INTO predictions (
        timestamp, filename, duration, model, emotion,
        confidence, confidence_pct, confidence_level, top_3,
        probabilities_json, features_json, user_notes, user_tag, is_starred
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, filename, float(duration), str(model), str(emotion).lower(),
        float(confidence), str(confidence_pct), str(confidence_level), str(top_3),
        prob_json, feat_json, user_notes, user_tag, int(is_starred)
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_predictions(
    emotion_filter: str = "ALL",
    model_filter: str = "ALL",
    search_query: str = "",
    sort_by: str = "Newest First",
    starred_only: bool = False,
    limit: int = 500,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve prediction records with filtering, search, and sorting (READ).
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params: List[Any] = []

    if emotion_filter and emotion_filter != "ALL":
        query += " AND LOWER(emotion) = ?"
        params.append(emotion_filter.lower())

    if model_filter and model_filter != "ALL":
        query += " AND UPPER(model) = ?"
        params.append(model_filter.upper())

    if search_query:
        query += " AND (filename LIKE ? OR user_notes LIKE ? OR user_tag LIKE ? OR emotion LIKE ?)"
        term = f"%{search_query}%"
        params.extend([term, term, term, term])

    if starred_only:
        query += " AND is_starred = 1"

    # Sorting
    if sort_by == "Newest First":
        query += " ORDER BY id DESC"
    elif sort_by == "Oldest First":
        query += " ORDER BY id ASC"
    elif sort_by == "Highest Confidence":
        query += " ORDER BY confidence DESC"
    elif sort_by == "Lowest Confidence":
        query += " ORDER BY confidence ASC"
    elif sort_by == "Duration":
        query += " ORDER BY duration DESC"
    else:
        query += " ORDER BY id DESC"

    query += f" LIMIT {int(limit)}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        item = dict(r)
        # Parse probabilities json
        try:
            item['probabilities'] = json.loads(item.get('probabilities_json', '{}'))
        except Exception:
            item['probabilities'] = {}
        # Parse features json
        try:
            item['features'] = json.loads(item.get('features_json', '{}'))
        except Exception:
            item['features'] = {}
        results.append(item)

    return results


def get_prediction_by_id(pred_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single prediction record by ID (READ)."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    item = dict(row)
    try:
        item['probabilities'] = json.loads(item.get('probabilities_json', '{}'))
    except Exception:
        item['probabilities'] = {}
    try:
        item['features'] = json.loads(item.get('features_json', '{}'))
    except Exception:
        item['features'] = {}
    return item


def update_prediction(
    pred_id: int,
    user_notes: Optional[str] = None,
    user_tag: Optional[str] = None,
    is_starred: Optional[int] = None,
    db_path: Optional[str] = None
) -> bool:
    """
    Update notes, tag, or starred flag of a prediction record (UPDATE).
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    updates = []
    params = []

    if user_notes is not None:
        updates.append("user_notes = ?")
        params.append(str(user_notes))

    if user_tag is not None:
        updates.append("user_tag = ?")
        params.append(str(user_tag))

    if is_starred is not None:
        updates.append("is_starred = ?")
        params.append(int(is_starred))

    if not updates:
        conn.close()
        return False

    params.append(pred_id)
    query = f"UPDATE predictions SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, params)
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()

    return rows_affected > 0


def delete_prediction(pred_id: int, db_path: Optional[str] = None) -> bool:
    """
    Delete a single prediction record by ID (DELETE).
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()

    return rows_affected > 0


def clear_all_predictions(db_path: Optional[str] = None) -> bool:
    """
    Delete all prediction records from the database (DELETE ALL).
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return True


def get_history_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute aggregate statistics for the history database.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), AVG(confidence) FROM predictions")
    count, avg_conf = cursor.fetchone()
    count = count or 0
    avg_conf = float(avg_conf or 0.0)

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_starred = 1")
    starred_count = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE confidence >= 0.70")
    high_conf_count = cursor.fetchone()[0] or 0

    cursor.execute("SELECT emotion, COUNT(*) FROM predictions GROUP BY emotion ORDER BY COUNT(*) DESC")
    emotion_counts = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT model, COUNT(*) FROM predictions GROUP BY model")
    model_counts = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    top_emotion = next(iter(emotion_counts.keys())) if emotion_counts else "None"

    return {
        "total_records": count,
        "average_confidence": round(avg_conf, 4),
        "average_confidence_pct": f"{avg_conf * 100:.1f}%",
        "starred_records": starred_count,
        "high_confidence_records": high_conf_count,
        "top_emotion": top_emotion,
        "emotion_breakdown": emotion_counts,
        "model_breakdown": model_counts
    }


def get_setting(key: str, default_val: str = "", db_path: Optional[str] = None) -> str:
    """Retrieve an application setting value."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()

    return row[0] if row else default_val


def set_setting(key: str, value: str, db_path: Optional[str] = None) -> None:
    """Store or update an application setting value."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO settings (key, value, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (key, str(value), now_str))

    conn.commit()
    conn.close()
