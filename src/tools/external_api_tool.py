"""External domain API tool adhering to universal response envelope (FR-401 - FR-403, FR-501 - FR-503)."""

from typing import List, Optional

from .external_api_client import ArXivClient
from .schemas import Citation, ToolResponse


class ExternalAPITool:
    """Agent tool for querying external domain research corpora (ArXiv scientific papers)."""

    name: str = "external_literature_tool"
    description: str = "Query academic literature and scientific research papers on ArXiv."

    def __init__(self, client: Optional[ArXivClient] = None):
        self.client = client or ArXivClient()

    def run(self, query: str, max_results: int = 3) -> str:
        """Queries ArXiv and returns serialized ToolResponse JSON."""
        try:
            result = self.client.search(query, max_results=max_results)
            status = result.get("status", "ERROR")

            if status == "ERROR":
                err_msg = result.get("error", "External API literature query failed.")
                resp = ToolResponse(
                    status="ERROR",
                    source_used="ARXIV",
                    answer=f"Academic literature API error: {err_msg}",
                    citations=[],
                    confidence=0.0,
                    details={"error": err_msg},
                )
                return resp.to_json()

            elif status == "INSUFFICIENT_CONTEXT":
                resp = ToolResponse(
                    status="INSUFFICIENT_CONTEXT",
                    source_used="ARXIV",
                    answer="No scientific papers or academic literature found for this query.",
                    citations=[],
                    confidence=0.0,
                    details={"papers_count": 0},
                )
                return resp.to_json()

            # Process OK results
            papers = result.get("papers", [])
            citations: List[Citation] = []
            summaries: List[str] = []

            for paper in papers:
                title = paper.get("title", "Research Paper")
                url = paper.get("url", "")
                authors = ", ".join(paper.get("authors", []))
                abstract = paper.get("abstract", "")

                citations.append(Citation(label=title, locator=url))
                summaries.append(
                    f"Title: {title}\nAuthors: {authors}\nURL: {url}\nAbstract: {abstract}"
                )

            resp = ToolResponse(
                status="OK",
                source_used="ARXIV",
                answer="\n\n".join(summaries),
                citations=citations,
                confidence=0.90,
                details={"papers_found": len(papers)},
            )
            return resp.to_json()

        except Exception as exc:
            resp = ToolResponse(
                status="ERROR",
                source_used="ARXIV",
                answer=f"External literature API unexpected failure: {str(exc)}",
                citations=[],
                confidence=0.0,
                details={"error": str(exc)},
            )
            return resp.to_json()

    def __call__(self, query: str) -> str:
        return self.run(query)
