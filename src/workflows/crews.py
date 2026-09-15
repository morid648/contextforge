"""Context-gathering crew and concurrent execution harness (FR-602)."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional
from crewai import Crew, Process

from ..tools.schemas import ToolResponse
from .agents import (
    create_external_api_agent,
    create_memory_agent,
    create_rag_agent,
    create_web_agent,
)
from .tasks import create_task


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


def create_context_gathering_crew(query: str, tools: Dict[str, Any]) -> Crew:
    """Instantiates a CrewAI Crew containing the 4 context agents and their tasks."""
    rag_agent = create_rag_agent(tools=[tools["rag_tool"]] if "rag_tool" in tools else None)
    memory_agent = create_memory_agent(tools=[tools["memory_tool"]] if "memory_tool" in tools else None)
    web_agent = create_web_agent(tools=[tools["web_tool"]] if "web_tool" in tools else None)
    ext_agent = create_external_api_agent(tools=[tools["external_api_tool"]] if "external_api_tool" in tools else None)

    tasks = [
        create_task("rag_task", rag_agent, query=query),
        create_task("memory_task", memory_agent, query=query),
        create_task("web_search_task", web_agent, query=query),
        create_task("external_api_task", ext_agent, query=query),
    ]

    return Crew(
        agents=[rag_agent, memory_agent, web_agent, ext_agent],
        tasks=tasks,
        verbose=True,
    )
