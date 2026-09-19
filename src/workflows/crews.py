"""Concurrent execution harness for the 4 heterogeneous context tools (FR-602)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from ..tools.schemas import ToolResponse


class ContextGatheringHarness:
    """Executes the 4 heterogeneous context tools concurrently (FR-602)."""

    def __init__(self, tools: Dict[str, Any]):
        self.rag_tool = tools.get("rag_tool")
        self.memory_tool = tools.get("memory_tool")
        self.web_tool = tools.get("web_tool")
        self.external_api_tool = tools.get("external_api_tool")

    def gather_parallel(self, query: str) -> Dict[str, ToolResponse]:
        """Runs all 4 context-gathering tools simultaneously in a thread pool."""
        results: Dict[str, ToolResponse] = {}

        tasks = {
            "RAG": lambda: self.rag_tool.run(query) if self.rag_tool else None,
            "MEMORY": lambda: self.memory_tool.run(query) if self.memory_tool else None,
            "WEB": lambda: self.web_tool.run(query) if self.web_tool else None,
            "ARXIV": lambda: self.external_api_tool.run(query) if self.external_api_tool else None,
        }

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_source = {
                executor.submit(fn): source_name for source_name, fn in tasks.items()
            }

            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    raw_json = future.result()
                    if raw_json:
                        parsed = ToolResponse.model_validate_json(raw_json)
                        results[source_name] = parsed
                    else:
                        results[source_name] = ToolResponse(
                            status="ERROR",
                            source_used=source_name if source_name in ["RAG", "MEMORY", "WEB", "ARXIV"] else "EXTERNAL_API",
                            answer=f"Tool '{source_name}' is not initialized.",
                            confidence=0.0,
                        )
                except Exception as exc:
                    results[source_name] = ToolResponse(
                        status="ERROR",
                        source_used=source_name if source_name in ["RAG", "MEMORY", "WEB", "ARXIV"] else "EXTERNAL_API",
                        answer=f"Tool execution failed: {str(exc)}",
                        confidence=0.0,
                    )

        return results
