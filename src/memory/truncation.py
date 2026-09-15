"""Intelligent sentence and word boundary truncation utility (FR-202)."""

import re


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncates text to max_length prioritizing sentence boundaries, then word boundaries."""
    text = text.strip()
    if len(text) <= max_length:
        return text

    # Slice candidate substring
    candidate = text[:max_length]

    # Look for sentence terminators (. ! ?)
    sentence_matches = list(re.finditer(r"[\.\!\?]\s+", candidate))
    if sentence_matches:
        last_match = sentence_matches[-1]
        end_idx = last_match.end() - 1  # include punctuation, omit trailing space
        return candidate[:end_idx].strip()

    # Fallback to nearest word boundary (space)
    last_space = candidate.rfind(" ")
    if last_space > int(max_length * 0.5):
        return candidate[:last_space].strip() + "..."

    # Hard fallback if no word boundary found
    return candidate.strip() + "..."
