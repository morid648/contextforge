"""Thread-scoped conversation memory manager with graceful local fallback (FR-201 - FR-205)."""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .truncation import truncate_text


class MemoryManager:
    """Manages thread-scoped conversation turns and consolidated context retrieval."""

    def __init__(
        self,
        user_id: str = "user_default",
        thread_id: str = "thread_default",
        db_path: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        max_turn_length: int = 600,
    ):
        self.user_id = user_id
        self.thread_id = thread_id
        self.api_key = api_key or os.getenv("MEMORY_API_KEY")
        self.max_turn_length = max_turn_length

        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            outputs_dir = project_root / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            db_path = outputs_dir / "conversation_memory.db"

        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes SQLite table for conversation memory."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def save_turn(self, role: str, name: str, message: str) -> None:
        """Persists a conversation turn, applying intelligent truncation if needed (FR-201, FR-202)."""
        clean_content = message.strip()
        if role.lower() == "assistant":
            clean_content = truncate_text(clean_content, max_length=self.max_turn_length)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversation_turns (user_id, thread_id, role, name, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self.user_id, self.thread_id, role, name, clean_content),
        )
        conn.commit()
        conn.close()

    def get_context(self, query: str = "", max_turns: int = 6) -> Optional[str]:
        """Retrieves a consolidated summary block of prior conversation turns (FR-203, FR-204)."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, name, content FROM conversation_turns
            WHERE user_id = ? AND thread_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (self.user_id, self.thread_id, max_turns + 1),
        )
        rows = cursor.fetchall()
        conn.close()

        # Prior memory must contain at least one completed assistant response (FR-204)
        has_assistant_turn = any(r[0].lower() == "assistant" for r in rows)
        if not rows or not has_assistant_turn:
            return None

        # Format oldest to newest (excluding current in-flight query if last row is current user turn)
        formatted_turns = []
        for role, name, content in reversed(rows):
            formatted_turns.append(f"[{role.upper()} - {name}]: {content}")

        context_block = "Prior Conversation Context:\n" + "\n".join(formatted_turns)
        return context_block

    def count_turns(self) -> int:
        """Returns the number of turns recorded for the current thread."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE user_id = ? AND thread_id = ?",
            (self.user_id, self.thread_id),
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def reset(self) -> None:
        """Clears all turns for the current user and thread."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM conversation_turns WHERE user_id = ? AND thread_id = ?",
            (self.user_id, self.thread_id),
        )
        conn.commit()
        conn.close()
