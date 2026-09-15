"""Evaluator task dynamic builder, schema validator, and error-isolation auditor (FR-603 - FR-605)."""

import json
import re
from typing import Any, Dict, List, Optional
from ..generation.schemas import ContextEvaluationResult
from ..tools.schemas import ToolResponse

# Patterns for queries whose intent is a document-level overview (summary, brief, etc.)
_SUMMARY_INTENT_RE = re.compile(
    r"\b(summar|brief|overview|abstract|outline|describe|what is this|what does this|"
    r"tell me about|explain this|give me a|what('s| is) in|content of|about this|this document|this paper|this file)",
    re.IGNORECASE,
)


class EvaluatorEngine:
    """Evaluates raw heterogeneous source outputs, scores relevance, and enforces error isolation."""

    def __init__(self, relevance_threshold: float = 0.4):
        self.relevance_threshold = relevance_threshold

    def evaluate_sources(
        self, query: str, raw_sources: Dict[str, ToolResponse]
    ) -> ContextEvaluationResult:
        """Audits all raw context sources for the query, enforcing error isolation (FR-605)."""
        relevance_scores: Dict[str, float] = {}
        filtered_context: Dict[str, Any] = {}
        relevant_sources: List[str] = []
        reasoning_lines: List[str] = []

        STOP_WORDS = {
            "what", "did", "does", "the", "a", "an", "in", "on", "of", "for",
            "is", "are", "about", "to", "and", "or", "show", "how", "with",
            "from", "by", "at", "it", "this", "that", "these", "those"
        }
        all_terms = re.findall(r"\w+", query.lower())
        query_terms = set(t for t in all_terms if t not in STOP_WORDS)
        if not query_terms:
            query_terms = set(all_terms)

        # Detect if the query is a document-level summary / overview intent.
        # Term-overlap scoring is useless for these ("summarize", "brief", etc. never
        # appear inside document body text), so we bypass it for OK-status sources.
        is_summary_query = bool(_SUMMARY_INTENT_RE.search(query))

        for source_name, tool_resp in raw_sources.items():
            # Error isolation rule: ERROR sources must NEVER be in relevant_sources (FR-605)
            if tool_resp.status == "ERROR":
                relevance_scores[source_name] = 0.0
                reasoning_lines.append(
                    f"Excluded '{source_name}': Tool encountered an error ({tool_resp.answer[:80]}...)."
                )
                continue

            if tool_resp.status == "INSUFFICIENT_CONTEXT":
                relevance_scores[source_name] = 0.0
                reasoning_lines.append(
                    f"Excluded '{source_name}': Tool reported insufficient context."
                )
                continue

            # Compute content relevance
            content = tool_resp.answer.lower()
            content_terms = set(re.findall(r"\w+", content))
            overlap = query_terms.intersection(content_terms)

            # For summary-intent queries: skip term-overlap penalty; score purely on
            # tool confidence so that populated sources (OK + non-empty answer) pass.
            if is_summary_query and tool_resp.status == "OK" and tool_resp.answer.strip():
                score = round(min(1.0, max(self.relevance_threshold, tool_resp.confidence)), 2)
            else:
                # Base relevance on term overlap and tool confidence
                overlap_ratio = len(overlap) / max(1, len(query_terms))
                score = round(min(1.0, 0.5 * overlap_ratio + 0.5 * tool_resp.confidence), 2)

            relevance_scores[source_name] = score

            if score >= self.relevance_threshold:
                relevant_sources.append(source_name)
                filtered_context[source_name] = {
                    "answer": tool_resp.answer,
                    "citations": [c.model_dump() for c in tool_resp.citations],
                    "confidence": tool_resp.confidence,
                }
                reasoning_lines.append(
                    f"Included '{source_name}': Relevant to query with score {score}."
                )
            else:
                reasoning_lines.append(
                    f"Excluded '{source_name}': Low relevance score ({score} < {self.relevance_threshold})."
                )

        reasoning = " ".join(reasoning_lines)
        return ContextEvaluationResult(
            relevant_sources=relevant_sources,
            filtered_context=filtered_context,
            relevance_scores=relevance_scores,
            reasoning=reasoning,
        )

    def parse_evaluation_output(self, raw_output: str) -> ContextEvaluationResult:
        """Parses LLM output into ContextEvaluationResult with robust fallback (FR-604)."""
        clean_text = raw_output.strip()

        # Try direct JSON parsing
        try:
            data = json.loads(clean_text)
            return ContextEvaluationResult.model_validate(data)
        except Exception:
            pass

        # Try markdown code block extraction
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return ContextEvaluationResult.model_validate(data)
            except Exception:
                pass

        # Robust best-effort fallback
        return ContextEvaluationResult(
            relevant_sources=[],
            filtered_context={},
            relevance_scores={},
            reasoning=f"Failed to parse LLM structured evaluation. Raw output: {clean_text[:200]}",
        )
