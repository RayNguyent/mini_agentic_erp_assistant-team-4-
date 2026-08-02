"""Merge specialist results into one answer.

Two rules: every specialist's contribution is attributed (the reviewer can see
which agent produced which part of the answer), and the union of citations
survives synthesis — a doc-dependent claim that loses its citation on the way
through the merge is exactly the failure the RAG citation gate exists to catch,
so this module never silently drops one.
"""

from app.state import AgentResult, Citation


def _render_erp_analyst(result: AgentResult) -> str:
    from app.runtime import _render_tool_output

    if result.tool_output is not None:
        return _render_tool_output(result.tool_used, result.tool_output)
    return result.error_message or "No project data available."


def _render_doc_researcher(result: AgentResult) -> str:
    return result.answer or "No relevant project documents were found."


def _render_risk_writer(result: AgentResult) -> str:
    if result.tool_output is not None:
        from app.runtime import _render_create_risk

        return _render_create_risk(result.tool_output)
    return result.error_message or "The risk action could not be completed."


_RENDERERS = {
    "erp_analyst": _render_erp_analyst,
    "doc_researcher": _render_doc_researcher,
    "risk_writer": _render_risk_writer,
}


def synthesize(results: list[AgentResult]) -> tuple[str, list[Citation], bool]:
    """Combine ordered specialist results.

    Returns `(answer, citations, all_failed)`. `all_failed` lets the caller
    decide whether to surface an error code — one specialist failing among
    several succeeding should degrade the answer, not error the whole request.
    """
    if not results:
        return "No result available.", [], True

    if len(results) == 1:
        result = results[0]
        renderer = _RENDERERS.get(result.agent)
        text = renderer(result) if renderer else (result.answer or "No result available.")
        return text, list(result.citations), not result.ok

    sections: list[str] = []
    citations: list[Citation] = []
    seen_citation_ids: set[str] = set()
    any_ok = False

    for result in results:
        any_ok = any_ok or result.ok
        renderer = _RENDERERS.get(result.agent)
        body = renderer(result) if renderer else (result.answer or "")
        label = result.agent.replace("_", " ").title()
        sections.append(f"**{label}**\n{body}")

        for citation in result.citations:
            if citation.chunk_id not in seen_citation_ids:
                citations.append(citation)
                seen_citation_ids.add(citation.chunk_id)

    return "\n\n".join(sections), citations, not any_ok
