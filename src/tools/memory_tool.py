"""Memory agent tool adhering to the universal response envelope (FR-204, FR-501 - FR-503)."""

from typing import Optional

from ..memory.memory import MemoryManager
from .schemas import Citation, ToolResponse


class MemoryTool:
    """Agent tool for querying session conversation history."""

    name: str = "memory_search_tool"
    description: str = "Retrieve prior conversation turns, user context, and preferences."

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()

    def run(self, query: str = "") -> str:
        """Retrieves conversational history and returns serialized ToolResponse JSON."""
        try:
            context = self.memory_manager.get_context(query)

            if not context or not context.strip():
                resp = ToolResponse(
                    status="INSUFFICIENT_CONTEXT",
                    source_used="MEMORY",
                    answer="No prior conversation turns recorded for this session thread.",
                    citations=[],
                    confidence=0.0,
                    details={"thread_id": self.memory_manager.thread_id},
                )
                return resp.to_json()

            citation = Citation(
                label="Conversation Session",
                locator=f"User: {self.memory_manager.user_id} | Thread: {self.memory_manager.thread_id}",
            )

            resp = ToolResponse(
                status="OK",
                source_used="MEMORY",
                answer=context,
                citations=[citation],
                confidence=0.85,
                details={"turns_available": self.memory_manager.count_turns()},
            )
            return resp.to_json()

        except Exception as exc:
            resp = ToolResponse(
                status="ERROR",
                source_used="MEMORY",
                answer=f"Memory retrieval error: {str(exc)}",
                citations=[],
                confidence=0.0,
                details={"error": str(exc)},
            )
            return resp.to_json()

    def __call__(self, query: str = "") -> str:
        return self.run(query)
