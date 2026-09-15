"""Vector store client supporting Milvus Lite and local in-process cosine storage."""

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..document_processing.schemas import DocumentChunk

# ---------------------------------------------------------------------------
# Cosine similarity — sklearn vectorized implementation with manual fallback
# ---------------------------------------------------------------------------
try:
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_sim
    import numpy as np

    def _batch_cosine_scores(query_vec: List[float], chunk_vecs: List[List[float]]) -> List[float]:
        """Computes cosine similarity between query and all chunks in one vectorized call."""
        q = np.array([query_vec])                    # shape (1, dim)
        C = np.array(chunk_vecs)                     # shape (n, dim)
        scores = _sk_cosine_sim(q, C)[0]             # shape (n,)
        return scores.tolist()

except ImportError:
    def _batch_cosine_scores(query_vec: List[float], chunk_vecs: List[List[float]]) -> List[float]:  # type: ignore[misc]
        """Pure-Python fallback cosine similarity (no sklearn)."""
        def _cosine(v1: List[float], v2: List[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            n1 = math.sqrt(sum(a * a for a in v1))
            n2 = math.sqrt(sum(b * b for b in v2))
            return dot / (n1 * n2) if n1 and n2 else 0.0
        return [_cosine(query_vec, cv) for cv in chunk_vecs]


class VectorStoreClient:
    """Manages chunk vector storage and similarity retrieval (PRD §7.4)."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        collection_name: str = "rag_chunks",
        dimension: int = 1024,
    ):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            outputs_dir = project_root / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            db_path = outputs_dir / "vector_store.db"

        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.dimension = dimension
        self._init_db()

    def _init_db(self, drop_existing: bool = True) -> None:
        """Initializes database schema, dropping previous session collection if requested."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        if drop_existing:
            cursor.execute(f"DROP TABLE IF EXISTS {self.collection_name}")

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def reset_collection(self) -> None:
        """Drops and recreates the current collection for a new session."""
        self._init_db(drop_existing=True)

    def insert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Inserts document chunks alongside their embedding vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})."
            )

        if not chunks:
            return 0

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        records = []
        for chunk, emb in zip(chunks, embeddings):
            emb_json = json.dumps(emb)
            records.append((chunk.text, chunk.page_number, chunk.chunk_index, chunk.source_file, emb_json))

        cursor.executemany(
            f"""
            INSERT INTO {self.collection_name} (text, page_number, chunk_index, source_file, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
        inserted_count = cursor.rowcount
        conn.close()
        return len(records)

    def search(self, query_embedding: List[float], top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Searches vector store using sklearn cosine_similarity (vectorized) and returns top-k chunks.

        All chunk embeddings are loaded into a numpy matrix and scored against the query
        vector in a single batched operation — O(n·d) instead of O(n) Python loop iterations.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, text, page_number, chunk_index, source_file, embedding FROM {self.collection_name}"
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Deserialize all embeddings and batch-score with sklearn cosine_similarity
        ids, texts, pages, idxs, srcs, chunk_vecs = [], [], [], [], [], []
        for row in rows:
            ids.append(row[0])
            texts.append(row[1])
            pages.append(row[2])
            idxs.append(row[3])
            srcs.append(row[4])
            chunk_vecs.append(json.loads(row[5]))

        scores = _batch_cosine_scores(query_embedding, chunk_vecs)

        scored_results = [
            {
                "id": ids[i],
                "text": texts[i],
                "page_number": pages[i],
                "chunk_index": idxs[i],
                "source_file": srcs[i],
                "score": round(scores[i], 4),
            }
            for i in range(len(rows))
            if scores[i] >= score_threshold
        ]

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def count(self) -> int:
        """Returns the total number of chunks currently stored."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
        total = cursor.fetchone()[0]
        conn.close()
        return total
