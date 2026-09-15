"""Web search agent tool adhering to the universal response envelope (FR-301 - FR-303, FR-501 - FR-503)."""

from typing import List, Optional

from .schemas import Citation, ToolResponse
from .web_search_client import WebSearchClient


class WebSearchTool:
    """Agent tool for searching live web sources and breaking news."""

    name: str = "web_search_tool"
    description: str = "Search the live web for recent developments, news, and external verification."

    def __init__(self, client: Optional[WebSearchClient] = None):
        self.client = client or WebSearchClient()

    def run(self, query: str, max_results: int = 3) -> str:
        """Executes a web search and returns serialized ToolResponse JSON."""
        try:
            result = self.client.search(query, max_results=max_results)
            status = result.get("status", "ERROR")

            if status == "ERROR":
                err_msg = result.get("error", "Web search failed or is unconfigured.")
                resp = ToolResponse(
                    status="ERROR",
                    source_used="WEB",
                    answer=f"Web search unavailable: {err_msg}",
                    citations=[],
                    confidence=0.0,
                    details={"error": err_msg},
                )
                return resp.to_json()

            elif status == "INSUFFICIENT_CONTEXT":
                resp = ToolResponse(
                    status="INSUFFICIENT_CONTEXT",
                    source_used="WEB",
                    answer="Web search executed successfully but found no relevant public results.",
                    citations=[],
                    confidence=0.0,
                    details={"results_count": 0},
                )
                return resp.to_json()

            # Process OK results
            items = result.get("results", [])
            citations: List[Citation] = []
            snippets: List[str] = []

            for item in items:
                title = item.get("title", "Web Result")
                url = item.get("url", "")
                snippet = item.get("snippet", "")
                citations.append(Citation(label=title, locator=url))
                snippets.append(f"[{title}]({url}):\n{snippet}")

            resp = ToolResponse(
                status="OK",
                source_used="WEB",
                answer="\n\n".join(snippets),
                citations=citations,
                confidence=0.80,
                details={"results_count": len(items)},
            )
            return resp.to_json()

        except Exception as exc:
            resp = ToolResponse(
                status="ERROR",
                source_used="WEB",
                answer=f"Web search unexpected error: {str(exc)}",
                citations=[],
                confidence=0.0,
                details={"error": str(exc)},
            )
            return resp.to_json()

    def __call__(self, query: str) -> str:
        return self.run(query)
