"""The 4-stage Research Assistant Flow state machine (PRD §4.1, FR-601 - FR-606)."""

from typing import Any, Dict, Optional

from ..generation.generation import GroundedSynthesizer
from ..generation.schemas import ContextEvaluationResult, FinalSynthesizedResponse
from ..memory.memory import MemoryManager
from ..rag.rag_pipeline import RAGPipeline
from ..tools.external_api_client import ArXivClient
from ..tools.external_api_tool import ExternalAPITool
from ..tools.memory_tool import MemoryTool
from ..tools.rag_tool import RAGTool
from ..tools.schemas import ToolResponse
from ..tools.web_search_client import WebSearchClient
from ..tools.web_search_tool import WebSearchTool
from .crews import ContextGatheringHarness
from .evaluator import EvaluatorEngine


class ResearchAssistantFlow:
    """Orchestrates the 4-stage context engineering research pipeline."""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_manager: Optional[MemoryManager] = None,
        web_search_client: Optional[WebSearchClient] = None,
        external_api_client: Optional[ArXivClient] = None,
        thread_id: Optional[str] = None,
    ):
        import uuid
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        if memory_manager is None:
            tid = thread_id or f"thread_{uuid.uuid4().hex[:8]}"
            self.memory_manager = MemoryManager(thread_id=tid)
        else:
            self.memory_manager = memory_manager

        self.web_client = web_search_client or WebSearchClient()
        self.external_api_client = external_api_client or ArXivClient()

        # Initialize tools
        self.tools = {
            "rag_tool": RAGTool(self.rag_pipeline),
            "memory_tool": MemoryTool(self.memory_manager),
            "web_tool": WebSearchTool(self.web_client),
            "external_api_tool": ExternalAPITool(self.external_api_client),
        }

        # Subsystems
        self.gatherer = ContextGatheringHarness(self.tools)
        self.evaluator = EvaluatorEngine(relevance_threshold=0.35)
        self.synthesizer = GroundedSynthesizer()

    def kickoff(self, query: str, user_name: str = "User") -> Dict[str, Any]:
        """Executes the full 4-stage pipeline."""
        # Stage 1: process_query - persist user turn to memory (FR-201)
        self.memory_manager.save_turn(role="user", name=user_name, message=query)

        # Stage 2: gather_context_from_all_sources - run 4 agents concurrently (FR-602)
        raw_sources: Dict[str, ToolResponse] = self.gatherer.gather_parallel(query)

        # Stage 3: evaluate_context_relevance - score & filter, isolate errors (FR-603 - FR-605)
        evaluation: ContextEvaluationResult = self.evaluator.evaluate_sources(query, raw_sources)

        # Stage 4: synthesize_final_response - generate grounded answer using only filtered context (FR-606)
        final_response: FinalSynthesizedResponse = self.synthesizer.synthesize(query, evaluation)

        # Persist assistant turn to memory (FR-201)
        self.memory_manager.save_turn(
            role="assistant", name="ResearchAssistant", message=final_response.answer
        )

        return {
            "status": final_response.status,
            "answer": final_response.answer,
            "citations": [c.model_dump() for c in final_response.citations],
            "confidence": final_response.confidence,
            "source_used": final_response.source_used,
            "evaluation": evaluation.model_dump(),
            "raw_sources": {k: v.model_dump() for k, v in raw_sources.items()},
        }
