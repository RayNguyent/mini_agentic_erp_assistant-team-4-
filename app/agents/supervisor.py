"""The supervisor: turns a request into a typed plan of specialist steps.

A plan is a list of AgentStep, each naming an agent and its inputs — never
free text the executor has to re-parse. The deterministic fallback reuses
app.runtime's classifier and keyword detection so the offline profile plans
exactly as sensibly as the LLM-backed path, just without a model call.
"""

import json
import logging

from app.rag.bm25 import tokenize
from app.runtime import READ_TOOLS, WRITE_TOOLS, default_classify
from app.state import AgentStep

logger = logging.getLogger(__name__)

# Vocabulary that signals "this needs the document corpus", distinct from the
# ERP-tool vocabulary already encoded in default_classify. A message hitting
# both triggers a fan-out plan rather than a single specialist.
_DOC_TERMS = frozenset(
    """
    policy governance charter document documents decision decisions rationale
    why sla agreement vendor contract retrospective lessons learned adr rule
    rules escalation definition scope purpose ownership approach cadence
    severity rating ratings close closes closed owner owns allowed permitted
    process procedure guideline guidelines requirement requirements
    """.split()
)

_PLANNER_SYSTEM = """Plan how to answer a project-delivery question using the \
available specialists.

- erp_analyst: live project data (status, tasks, sprint progress, budget, risks).
- doc_researcher: policy/rationale/history from project documents, with citations.
- risk_writer: creates a risk record (always requires human approval).

Return JSON only:
{"steps": [{"agent": "erp_analyst", "rationale": "...",
            "inputs": {"intent": "project_status", "tool_input": {"project_code": "PRJ-001"}}}]}

Use erp_analyst inputs.intent from: project_status, list_risks, list_project_tasks,
sprint_progress, budget_summary. Use doc_researcher inputs.question for the
sub-question it should research. Use risk_writer inputs.tool_input for
{project_code, risk_payload}. Plan multiple steps only when the question
genuinely needs more than one specialist's evidence to answer."""


def _mentions_doc_terms(message: str) -> bool:
    return bool(set(tokenize(message)) & _DOC_TERMS)


def plan_deterministic(message: str, history: list[dict] | None = None) -> list[AgentStep]:
    """Keyword-driven planning: the same intent detection the single-agent
    graph uses, extended with a check for document-shaped vocabulary."""
    intent, tool_input = default_classify(message, history)
    doc_signal = _mentions_doc_terms(message)

    if intent in WRITE_TOOLS:
        return [
            AgentStep(
                agent="risk_writer",
                rationale=f"'{intent}' is a write action and always requires approval.",
                inputs={"tool_input": tool_input},
            )
        ]

    if intent in READ_TOOLS:
        step = AgentStep(
            agent="erp_analyst",
            rationale=f"'{intent}' matches a live ERP data lookup.",
            inputs={"intent": intent, "tool_input": tool_input},
        )
        if doc_signal:
            # Fan-out: the question also names policy/rationale-shaped
            # vocabulary a document, not the ERP system, would answer.
            return [
                step,
                AgentStep(
                    agent="doc_researcher",
                    rationale="question also references document-shaped context.",
                    inputs={"question": message, "project_code": tool_input.get("project_code")},
                ),
            ]
        return [step]

    # No ERP tool matched. Route to the document researcher rather than the
    # bare conversational fallback whenever the message looks like a real
    # question — an empty plan should be rare, not the default.
    return [
        AgentStep(
            agent="doc_researcher",
            rationale="no ERP tool matched; treating as a document question.",
            inputs={"question": message},
        )
    ]


def _extract_json(raw: str) -> dict:
    import re

    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


_KNOWN_AGENTS = frozenset({"erp_analyst", "doc_researcher", "risk_writer"})


def plan(
    message: str,
    history: list[dict] | None = None,
    provider=None,
) -> list[AgentStep]:
    """The typed plan the supervisor produces. Never raises — a malformed or
    unavailable LLM response falls back to the deterministic planner, which is
    exactly the same guarantee the single-agent classifier already makes."""
    fallback = plan_deterministic(message, history)
    if provider is None:
        return fallback

    try:
        raw = provider.generate(message, system=_PLANNER_SYSTEM, history=history)
        parsed = _extract_json(raw)
        steps = [
            AgentStep(
                agent=item["agent"],
                rationale=str(item.get("rationale", "")),
                inputs=dict(item.get("inputs") or {}),
            )
            for item in parsed.get("steps", [])
            if item.get("agent") in _KNOWN_AGENTS
        ]
        return steps or fallback
    except Exception as exc:  # noqa: BLE001 - plan with the deterministic path
        logger.debug("LLM planning failed, using deterministic plan: %s", exc)
        return fallback
