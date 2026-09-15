"""Collapsible sources and citations drawer with resilient error fallback (FR-703, FR-705)."""

from typing import Any, Dict
import streamlit as st


def render_citations_drawer(turn_data: Dict[str, Any]) -> None:
    """Renders the sources, evaluation scores, and citation cards for an assistant turn."""
    try:
        evaluation = turn_data.get("evaluation", {})
        raw_sources = turn_data.get("raw_sources", {})
        used_sources = turn_data.get("source_used", "NONE")
        confidence = turn_data.get("confidence", 0.0)

        with st.expander(f"🔍 Sources & Citations — [{used_sources}] (Confidence: {int(confidence * 100)}%)", expanded=False):
            # Evaluator Summary
            st.markdown("#### 🧠 Context Evaluator Audit")
            reasoning = evaluation.get("reasoning", "No evaluator reasoning provided.")
            st.info(f"**Evaluator Rationale:** {reasoning}")

            # Relevance Scores Metric Row
            scores = evaluation.get("relevance_scores", {})
            if scores:
                cols = st.columns(len(scores))
                for col, (source_name, score) in zip(cols, scores.items()):
                    with col:
                        st.metric(label=f"Relevance ({source_name})", value=f"{int(score * 100)}%")

            st.markdown("---")
            st.markdown("#### 📦 Source Inspectability Cards")

            # Render each source card
            for source_name, source_data in raw_sources.items():
                status = source_data.get("status", "UNKNOWN")
                answer = source_data.get("answer", "")
                citations = source_data.get("citations", [])
                conf = source_data.get("confidence", 0.0)

                # Colored status badge
                if status == "OK":
                    status_badge = "🟢 **STATUS: OK**"
                elif status == "INSUFFICIENT_CONTEXT":
                    status_badge = "🟡 **STATUS: INSUFFICIENT CONTEXT**"
                elif status == "ERROR":
                    status_badge = "🔴 **STATUS: ERROR**"
                else:
                    status_badge = f"⚪ **STATUS: {status}**"

                with st.expander(f"Source: {source_name} — {status_badge}", expanded=(status == "OK")):
                    st.markdown(f"**Confidence:** `{conf}`")
                    st.markdown("**Content Excerpt:**")
                    st.text(answer[:800] + ("..." if len(answer) > 800 else ""))

                    if citations:
                        st.markdown("**Pinpoint Citations:**")
                        for cit in citations:
                            label = cit.get("label", "Source")
                            loc = cit.get("locator", "N/A")
                            if loc.startswith("http"):
                                st.markdown(f"- [{label}]({loc})")
                            else:
                                st.markdown(f"- **{label}**: `{loc}`")

    except Exception as render_err:
        # Fallback debug expander to prevent UI crashes (FR-705)
        with st.expander("⚠️ Raw Citation Telemetry (Debug Fallback)", expanded=False):
            st.error(f"Failed to render formatted citations: {str(render_err)}")
            st.json(turn_data)
