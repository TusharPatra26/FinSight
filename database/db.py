"""SQLite persistence for analysis history. All queries are parameterised (no string-built SQL)."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    TEXT,
    source      TEXT NOT NULL CHECK (source IN ('single', 'batch')),
    input_text  TEXT NOT NULL,
    sentiment   TEXT NOT NULL,
    score       REAL NOT NULL,
    score_type  TEXT NOT NULL,
    margin      REAL,
    margin_band TEXT,
    key_terms   TEXT,
    model_id    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_batch ON analyses (batch_id);
"""
_COLUMNS = ("batch_id", "source", "input_text", "sentiment", "score", "score_type", "margin", "margin_band", "key_terms", "model_id", "created_at")


@contextmanager
def connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_from_result(result: dict, source: str, batch_id, created_at: str) -> tuple:
    terms = ", ".join(t["term"] for t in result["explanation"]["supporting"][:3])
    return (batch_id, source, result["text"], result["sentiment"], result["score"], result["score_type"],
            result["margin"], result["margin_band"], terms or None, result["model_id"], created_at)


def add_results(db_path, results, source: str, batch_id=None) -> int:
    """Store prediction results (dicts from SentimentModel) in one transaction. Returns rows written."""
    ts = _now()
    rows = [_row_from_result(r, source, batch_id, ts) for r in results]
    with connect(db_path) as conn:
        conn.executemany(f"INSERT INTO analyses ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})", rows)
    return len(rows)


def _where(sentiment=None, source=None, query=None, batch_id=None):
    clauses, params = [], []
    if sentiment:
        clauses.append("sentiment = ?"); params.append(sentiment)
    if source:
        clauses.append("source = ?"); params.append(source)
    if batch_id:
        clauses.append("batch_id = ?"); params.append(batch_id)
    if query:
        like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        clauses.append("input_text LIKE ? ESCAPE '\\'"); params.append(like)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def list_analyses(db_path, limit=50, offset=0, sentiment=None, source=None, query=None, batch_id=None, newest_first=True) -> list:
    where, params = _where(sentiment, source, query, batch_id)
    order = "DESC" if newest_first else "ASC"
    sql = f"SELECT * FROM analyses{where} ORDER BY created_at {order}, id {order} LIMIT ? OFFSET ?"
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, params + [int(limit), int(offset)])]


def count_analyses(db_path, sentiment=None, source=None, query=None, batch_id=None) -> int:
    where, params = _where(sentiment, source, query, batch_id)
    with connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM analyses{where}", params).fetchone()[0]


def sentiment_counts(db_path) -> dict:
    with connect(db_path) as conn:
        return {r["sentiment"]: r["n"] for r in conn.execute("SELECT sentiment, COUNT(*) AS n FROM analyses GROUP BY sentiment")}


def daily_counts(db_path, days: int = 30) -> list:
    """Analyses per UTC day and sentiment for the most recent `days` days that have data."""
    sql = ("SELECT substr(created_at, 1, 10) AS day, sentiment, COUNT(*) AS n FROM analyses "
           "GROUP BY day, sentiment ORDER BY day DESC LIMIT ?")
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, (int(days) * 3,))][::-1]


def clear_history(db_path) -> int:
    """Delete ALL history rows. Callers must confirm with the user first."""
    with connect(db_path) as conn:
        return conn.execute("DELETE FROM analyses").rowcount
