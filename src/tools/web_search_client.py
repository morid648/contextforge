"""Web search client with auto-detection for Tavily and WebSearchAPI.ai (FR-301, FR-302, FR-303)."""

import os
from typing import Any, Dict, List, Optional
import requests


class WebSearchClient:
    """Queries web search APIs with auto-detection and graceful failure modes.

    Supported providers (auto-detected by key prefix):
        - Firecrawl        — key prefix: fc-
        - WebSearchAPI.ai  — key prefix: wsa_
        - Tavily           — key prefix: tvly-
    """

    def __init__(self, api_key: Optional[str] = None, max_snippet_length: int = 350):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("FIRECRAWL_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")
        self.max_snippet_length = max_snippet_length
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Auto-detects the web search provider from the API key prefix."""
        if not self.api_key:
            return "unknown"
        if self.api_key.startswith("fc-"):
            return "firecrawl"
        if self.api_key.startswith("wsa_"):
            return "websearchapi"
        if self.api_key.startswith("tvly-"):
            return "tavily"
        # Default fallback: try Tavily format
        return "tavily"

    def is_available(self) -> bool:
        """Checks if web search has an active API key configured."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    def search(self, query: str, max_results: int = 4) -> Dict[str, Any]:
        """Performs a web search, returning normalized results or an error payload."""
        if not self.is_available():
            return {
                "status": "ERROR",
                "error": "WEB_SEARCH_API_KEY is not configured.",
                "results": [],
            }

        if self._provider == "firecrawl":
            return self._search_firecrawl(query, max_results)
        if self._provider == "websearchapi":
            return self._search_websearchapi(query, max_results)
        return self._search_tavily(query, max_results)

    # ------------------------------------------------------------------
    # Firecrawl provider
    # ------------------------------------------------------------------

    def _search_firecrawl(self, query: str, max_results: int) -> Dict[str, Any]:
        """Queries Firecrawl search endpoint: POST https://api.firecrawl.dev/v1/search."""
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v1/search",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "limit": max_results,
                },
                timeout=12,
            )

            if response.status_code != 200:
                return {
                    "status": "ERROR",
                    "error": f"Firecrawl returned HTTP {response.status_code}: {response.text[:200]}",
                    "results": [],
                }

            data = response.json()
            raw_results = data.get("data") or data.get("results") or []

            if not raw_results:
                return {"status": "INSUFFICIENT_CONTEXT", "error": None, "results": []}

            return {
                "status": "OK",
                "error": None,
                "results": self._normalize(raw_results),
            }

        except Exception as exc:
            return {
                "status": "ERROR",
                "error": f"Firecrawl exception: {str(exc)}",
                "results": [],
            }

    # ------------------------------------------------------------------
    # WebSearchAPI.ai provider
    # ------------------------------------------------------------------

    def _search_websearchapi(self, query: str, max_results: int) -> Dict[str, Any]:
        """Queries WebSearchAPI.ai — Authorization: Bearer header, JSON response."""
        try:
            response = requests.post(
                "https://api.websearchapi.ai/search",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "max_results": max_results,
                    "include_content": True,
                    "language": "en",
                },
                timeout=10,
            )

            if response.status_code != 200:
                return {
                    "status": "ERROR",
                    "error": f"WebSearchAPI.ai returned HTTP {response.status_code}: {response.text[:200]}",
                    "results": [],
                }

            data = response.json()
            # Response shape: {"results": [...]} or {"organic": [...]}
            raw_results = data.get("results") or data.get("organic") or []

            if not raw_results:
                return {"status": "INSUFFICIENT_CONTEXT", "error": None, "results": []}

            return {
                "status": "OK",
                "error": None,
                "results": self._normalize(raw_results),
            }

        except Exception as exc:
            return {
                "status": "ERROR",
                "error": f"WebSearchAPI.ai exception: {str(exc)}",
                "results": [],
            }

    # ------------------------------------------------------------------
    # Tavily provider
    # ------------------------------------------------------------------

    def _search_tavily(self, query: str, max_results: int) -> Dict[str, Any]:
        """Queries Tavily search API — api_key in request body."""
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                },
                timeout=8,
            )

            if response.status_code != 200:
                return {
                    "status": "ERROR",
                    "error": f"Tavily returned HTTP {response.status_code}: {response.text[:200]}",
                    "results": [],
                }

            raw_results = response.json().get("results", [])
            if not raw_results:
                return {"status": "INSUFFICIENT_CONTEXT", "error": None, "results": []}

            return {
                "status": "OK",
                "error": None,
                "results": self._normalize(raw_results),
            }

        except Exception as exc:
            return {
                "status": "ERROR",
                "error": f"Tavily exception: {str(exc)}",
                "results": [],
            }

    # ------------------------------------------------------------------
    # Normalizer — maps provider-specific shapes to a common schema
    # ------------------------------------------------------------------

    def _normalize(self, raw: List[Dict]) -> List[Dict]:
        """Normalizes heterogeneous result shapes to {title, url, snippet}."""
        normalized = []
        for item in raw:
            snippet = (
                item.get("content")
                or item.get("snippet")
                or item.get("description")
                or item.get("markdown")
                or item.get("body")
                or ""
            )
            if len(snippet) > self.max_snippet_length:
                snippet = snippet[: self.max_snippet_length].strip() + "..."

            normalized.append({
                "title": item.get("title") or item.get("name") or "Web Source",
                "url": item.get("url") or item.get("link") or "",
                "snippet": snippet,
            })
        return normalized
