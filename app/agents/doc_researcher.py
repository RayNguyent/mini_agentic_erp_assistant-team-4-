"""The DocResearcher specialist: project-document RAG.

Thin by design — the real work (retrieve, grade, refine, verify) is the
agentic loop in app/rag/agentic.py, which is unit-tested on its own. This
module's only job is the specialist-agent contract: allowlist (no tools),
required permission (document read), and mapping RagState onto AgentResult.
"""

from app.agents.base import authorize, denied, timed
from app.errors import ErrorCode
from app.graph.engine import GraphContext
from app.rag.agentic import run_agentic_rag
from app.state import AgentResult, AgentState, AgentStep


class DocResearcher:
    name = "doc_researcher"
    description = "Answers questions from project documents, with citations."
    allowed_tools: frozenset[str] = frozenset()  # no ERP tool access at all
    required_permission = "document.read"

    def run(self, step: AgentStep, state: AgentState, ctx: GraphContext) -> AgentResult:
        if not authorize(self, state, ctx):
            return denied(self.name, self.required_permission, state.actor)

        def _run() -> AgentResult:
            retriever = ctx.require("retriever")
            question = step.inputs.get("question") or ctx.require("message")
            role = state.actor.role if state.actor else None

            rag_state = run_agentic_rag(
                question,
                retriever,
                role=role,
                project_code=step.inputs.get("project_code"),
                provider=ctx.get("provider"),
                on_status=ctx.get("on_status"),
                trace=ctx.trace,
            )

            if rag_state.refused:
                return AgentResult(
                    agent=self.name,
                    ok=True,
                    answer=rag_state.answer,
                    citations=[],
                    error_code=None,
                    context_log=rag_state.context_log,
                )

            return AgentResult(
                agent=self.name,
                ok=True,
                answer=rag_state.answer,
                citations=rag_state.citations,
                context_log=rag_state.context_log,
            )

        return timed(self.name, _run)
