"""Supervisor + specialist multi-agent layer, built on app.graph.

    Supervisor (typed plan) -> ERPAnalyst | DocResearcher | RiskWriter -> Synthesizer

Each specialist is confined by a tool allowlist and a required permission
(app/agents/base.py), which is what makes "the document researcher cannot
create a risk" a structural property rather than a prompt instruction.
"""

from app.agents.base import ScopedRegistry, SpecialistAgent, authorize, denied, timed
from app.agents.doc_researcher import DocResearcher
from app.agents.erp_analyst import ERPAnalyst
from app.agents.graph import (
    build_multi_agent_graph,
    default_agent_registry,
    resume_multi_agent,
    run_multi_agent,
)
from app.agents.risk_writer import RiskWriter
from app.agents.supervisor import plan
from app.agents.synthesizer import synthesize

__all__ = [
    "DocResearcher",
    "ERPAnalyst",
    "RiskWriter",
    "ScopedRegistry",
    "SpecialistAgent",
    "authorize",
    "build_multi_agent_graph",
    "default_agent_registry",
    "denied",
    "plan",
    "resume_multi_agent",
    "run_multi_agent",
    "synthesize",
    "timed",
]
