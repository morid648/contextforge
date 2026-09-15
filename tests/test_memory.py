"""Tests for conversation memory and truncation."""

from src.memory.truncation import truncate_text
from src.memory.memory import MemoryManager


def test_truncate_text_sentence_boundary():
    """Verify truncation at sentence boundaries."""
    text = "First sentence here. Second sentence follows. Third sentence is much longer and keeps going."
    truncated = truncate_text(text, max_length=50)
    assert truncated.endswith(".")
    assert "First sentence here." in truncated
    assert "Third sentence" not in truncated


def test_truncate_text_word_boundary():
    """Verify truncation falls back to word boundary if no sentence boundary."""
    text = "A very long sentence without any punctuation marks in between that extends beyond fifty chars"
    truncated = truncate_text(text, max_length=50)
    assert truncated.endswith("...")
    assert len(truncated) <= 50 + 3


def test_truncate_text_short_text():
    """Verify short text is returned unmodified."""
    text = "Short text."
    assert truncate_text(text, max_length=50) == "Short text."


def test_memory_manager_thread_isolation(tmp_path):
    """Verify user/thread isolation in memory manager."""
    db_file = tmp_path / "test_mem.db"
    mem_a = MemoryManager(user_id="user1", thread_id="thread_A", db_path=db_file)
    mem_b = MemoryManager(user_id="user1", thread_id="thread_B", db_path=db_file)

    # Empty context check (FR-204)
    assert mem_a.get_context("query") is None
    assert mem_b.get_context("query") is None

    # Save turn to thread A
    mem_a.save_turn(role="user", name="Alice", message="What are the margins?")
    mem_a.save_turn(role="assistant", name="Assistant", message="The margins are 24.5%.")

    # Thread A should have context, Thread B should still be None
    assert mem_a.count_turns() == 2
    assert mem_b.count_turns() == 0
    assert mem_b.get_context("query") is None

    ctx_a = mem_a.get_context("query")
    assert ctx_a is not None
    assert "Alice" in ctx_a
    assert "24.5%" in ctx_a
