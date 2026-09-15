"""Grounded synthesizer engine and response generator (FR-606, G2, PRD §7.3)."""

import json
import os
import re
from typing import List, Optional

from .schemas import ContextEvaluationResult, FinalSynthesizedResponse
from ..tools.schemas import Citation


class GroundedSynthesizer:
    """Generates strictly grounded answers from verified filtered context, enforcing anti-hallucination.

    LLM backend priority:
        1. Google Gemini  — gemini-1.5-flash (GEMINI_API_KEY)
        2. Deterministic  — extractive template synthesis (no API key needed)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL", model)

    def synthesize(
        self, query: str, evaluation: ContextEvaluationResult
    ) -> FinalSynthesizedResponse:
        """Synthesizes a response strictly adhering to filtered context (FR-606)."""
        # Grounding Rule 1: If no relevant sources passed evaluation, return INSUFFICIENT_CONTEXT (G2)
        if not evaluation.relevant_sources or not evaluation.filtered_context:
            return FinalSynthesizedResponse(
                status="INSUFFICIENT_CONTEXT",
                source_used="NONE",
                answer=(
                    "I cannot answer this query based on the verified context. "
                    "The uploaded documents, memory, and external search did not contain sufficient "
                    f"information regarding: '{query}'."
                ),
                citations=[],
                confidence=0.0,
                missing=[f"Factual data or documentation addressing: '{query}'"],
            )

        # Collect citations and passages from relevant sources
        all_citations: List[Citation] = []
        context_blocks: List[str] = []

        for src in evaluation.relevant_sources:
            src_data = evaluation.filtered_context.get(src, {})
            text = src_data.get("answer", "")
            raw_cits = src_data.get("citations", [])

            for cit in raw_cits:
                all_citations.append(Citation.model_validate(cit))

            context_blocks.append(f"=== SOURCE: {src} ===\n{text}")

        combined_context = "\n\n".join(context_blocks)

        # If Gemini API key is present, perform LLM-grounded generation
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.api_key)
                prompt = (
                    f"You are a strict, grounded research synthesizer. "
                    f"Answer the user query using ONLY the verified context below.\n"
                    f"Do NOT speculate, do NOT use outside memory, and cite each statement.\n\n"
                    f"User Query: {query}\n\n"
                    f"Verified Context:\n{combined_context}\n\n"
                    f"Answer in clear markdown with citations in brackets."
                )
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        system_instruction=(
                            "You are a factual, grounded research assistant. "
                            "Never hallucinate. Only use the context provided."
                        ),
                    ),
                )
                answer_text = response.text or ""
                return FinalSynthesizedResponse(
                    status="OK",
                    source_used=" + ".join(evaluation.relevant_sources),
                    answer=answer_text,
                    citations=all_citations,
                    confidence=0.92,
                    missing=[],
                )
            except Exception as e:
                print(f"[GroundedSynthesizer] Gemini LLM call fallback: {e}")

        # Extractive deterministic synthesis fallback (no API key required)
        summary_paragraphs = []
        for src in evaluation.relevant_sources:
            src_data = evaluation.filtered_context[src]
            ans = src_data.get("answer", "").strip()
            summary_paragraphs.append(f"**From {src}:**\n{ans}")

        grounded_answer = (
            f"Based on the verified context for '{query}':\n\n"
            + "\n\n".join(summary_paragraphs)
        )

        return FinalSynthesizedResponse(
            status="OK",
            source_used=" + ".join(evaluation.relevant_sources),
            answer=grounded_answer,
            citations=all_citations,
            confidence=0.88,
            missing=[],
        )
