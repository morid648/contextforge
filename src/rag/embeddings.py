"""Embeddings client supporting Voyage AI, Google Gemini, sklearn TF-IDF, and hash-based local fallback."""


import hashlib
import math
import os
from typing import List, Optional


class EmbeddingsClient:
    """Wrapper for generating contextualized document and query vector embeddings."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: int = 1024,
    ):
        self.provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "auto").lower()
        self.model = model or os.getenv("EMBEDDINGS_MODEL", "voyage-context-3")
        self.voyage_api_key = api_key or os.getenv("VOYAGE_API_KEY") or os.getenv("EMBEDDINGS_API_KEY")
        self.gemini_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.dimension = dimension

        # sklearn TF-IDF vectorizer — fitted lazily during embed_documents(), reused for embed_query()
        self._tfidf_vectorizer = None

        # Determine active backend
        if self.provider == "local_fallback":
            self._backend = "local_fallback"
        elif self.provider == "sklearn_tfidf":
            self._backend = "sklearn_tfidf"
        elif (self.provider == "voyage" or self.provider == "auto") and self.voyage_api_key:
            self._backend = "voyage"
        elif (self.provider == "gemini" or self.provider == "auto") and self.gemini_api_key:
            self._backend = "gemini"
            self.dimension = 768   # text-embedding-004 output dimension
        else:
            # Prefer sklearn TF-IDF over hash-based; fall back only if sklearn not installed
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
                self._backend = "sklearn_tfidf"
            except ImportError:
                self._backend = "local_fallback"

    def embed_documents(self, chunks: List[str]) -> List[List[float]]:
        """Generates contextualized embeddings for a batch of document chunks."""
        if not chunks:
            return []

        if self._backend == "voyage":
            try:
                import voyageai
                vo = voyageai.Client(api_key=self.voyage_api_key)
                result = vo.contextualized_embed(
                    inputs=[chunks],
                    model=self.model,
                )
                return result.embeddings[0]
            except Exception as e:
                # Degrade to local fallback if API call fails
                print(f"[EmbeddingsClient] Voyage API error ({e}), using local deterministic fallback.")
                return [self._generate_deterministic_vector(chunk) for chunk in chunks]

        elif self._backend == "gemini":
            try:
                from google import genai as ggenai
                client = ggenai.Client(api_key=self.gemini_api_key)
                result = client.models.embed_content(
                    model="text-embedding-004",
                    contents=chunks,
                )
                # result.embeddings is a list of ContentEmbedding objects
                return [e.values for e in result.embeddings]
            except Exception as e:
                print(f"[EmbeddingsClient] Gemini embed_documents error ({e}), using sklearn TF-IDF fallback.")
                return self._sklearn_embed_documents(chunks)

        elif self._backend == "sklearn_tfidf":
            return self._sklearn_embed_documents(chunks)

        return [self._generate_deterministic_vector(chunk) for chunk in chunks]

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a user query."""
        if self._backend == "voyage":
            try:
                import voyageai
                vo = voyageai.Client(api_key=self.voyage_api_key)
                result = vo.embed(
                    texts=[query],
                    model=self.model,
                    input_type="query",
                )
                return result.embeddings[0]
            except Exception as e:
                print(f"[EmbeddingsClient] Voyage query API error ({e}), using local fallback.")
                return self._generate_deterministic_vector(query)

        elif self._backend == "gemini":
            try:
                from google import genai as ggenai
                client = ggenai.Client(api_key=self.gemini_api_key)
                result = client.models.embed_content(
                    model="text-embedding-004",
                    contents=[query],
                )
                return result.embeddings[0].values
            except Exception as e:
                print(f"[EmbeddingsClient] Gemini embed_query error ({e}), using sklearn fallback.")
                return self._sklearn_embed_query(query)

        elif self._backend == "sklearn_tfidf":
            return self._sklearn_embed_query(query)

        return self._generate_deterministic_vector(query)

    def _sklearn_embed_documents(self, chunks: List[str]) -> List[List[float]]:
        """Fits a TF-IDF vectorizer on the document corpus and returns dense float vectors.

        Vectors are zero-padded to self.dimension so dimension is always consistent,
        even when the corpus vocabulary is smaller than max_features.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=self.dimension,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = self._tfidf_vectorizer.fit_transform(chunks)  # sparse (n, vocab_size)
        dense = matrix.toarray()                               # (n, vocab_size)

        # Pad to self.dimension if vocabulary < max_features
        n, vocab_size = dense.shape
        if vocab_size < self.dimension:
            import numpy as np
            dense = np.pad(dense, ((0, 0), (0, self.dimension - vocab_size)))

        return dense.tolist()

    def _sklearn_embed_query(self, query: str) -> List[float]:
        """Transforms a query using the already-fitted TF-IDF vectorizer.

        Falls back to fitting on the query itself (cold-start) and zero-pads to self.dimension.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=self.dimension,
                sublinear_tf=True,
            )
            self._tfidf_vectorizer.fit([query])

        vec = self._tfidf_vectorizer.transform([query]).toarray()[0]  # (vocab_size,)

        # Pad to self.dimension if vocabulary < max_features
        if len(vec) < self.dimension:
            import numpy as np
            vec = np.pad(vec, (0, self.dimension - len(vec)))

        return vec.tolist()

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """Generates a normalized deterministic dense vector using MD5/SHA256 projection."""
        vector: List[float] = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            return vector

        for word in words:
            # Hash word to seed index and value
            h = hashlib.sha256(word.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dimension
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            weight = (int(h[10:14], 16) / 65535.0) * sign
            vector[idx] += weight

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector
