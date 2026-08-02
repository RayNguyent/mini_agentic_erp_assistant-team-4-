"""Golden evaluation runner.

Runs every case in a dataset against the real system (real tool registry,
real RAG index, real multi-agent graph, real permission checks — nothing
mocked) and produces a versioned report. Deterministic checks only; there is
no rubric/LLM-judge scoring here because every case in cases.v1.json has an
objectively checkable acceptance condition (a citation id was returned, an
error code matches, a decision's action matches).

Usage:
    uv run python -m eval.runner
    uv run python -m eval.runner --dataset eval/golden/cases.v1.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.agents.graph import default_agent_registry, resume_multi_agent, run_multi_agent
from app.errors import ToolTimeoutError
from app.memory.long_term import LongTermMemory, MemoryEntry
from app.memory.policy import MemoryCandidate, decide
from app.rag.retrieve import build_retriever
from app.security.permissions import check as check_permission
from app.state import Actor, NextAction
from app.tools.registry import build_default_registry

DATASET_PATH = Path("eval/golden/cases.v1.json")
REPORTS_DIR = Path("eval/reports")
OVERALL_PASS_THRESHOLD = 0.80
CRITICAL_PASS_THRESHOLD = 1.00


@dataclass
class CaseResult:
    id: str
    family: str
    passed: bool
    acceptance_rule: str
    reasons: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def _flaky_registry(base_registry, tool_name: str, fail_times: int):
    """Wraps one tool so it raises a retryable error `fail_times` times
    before delegating to the real implementation — the deterministic way to
    prove the bounded-retry path actually retries, without depending on a
    real provider ever being flaky during a test run."""

    calls = {"n": 0}
    real_fn = base_registry.get(tool_name)

    def flaky_fn(tool_input: dict) -> dict:
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ToolTimeoutError(f"eval-injected transient failure #{calls['n']}")
        return real_fn(tool_input)

    class _WrappedRegistry:
        def get(self, name: str):
            return flaky_fn if name == tool_name else base_registry.get(name)

    return _WrappedRegistry()


class Harness:
    """Builds the real dependency graph once per run."""

    def __init__(self) -> None:
        self.tool_registry = build_default_registry()
        self.retriever = build_retriever()
        self.agents = default_agent_registry()

    def actor(self, role: str | None) -> Actor | None:
        return Actor(user_id="eval-runner", role=role) if role else None

    def run_chat(self, prompt: str, role: str | None, tool_registry=None):
        return run_multi_agent(
            prompt,
            tool_registry=tool_registry or self.tool_registry,
            retriever=self.retriever,
            actor=self.actor(role),
            agents=self.agents,
            authorize=check_permission,
        )


def _contains_all(haystack: str, needles: list[str]) -> list[str]:
    return [n for n in needles if n.lower() not in haystack.lower()]


def check_chat(case: dict, harness: Harness) -> CaseResult:
    expect = case["expect"]
    reasons: list[str] = []
    started = time.perf_counter()

    state = harness.run_chat(case["prompt"], case.get("role"))

    if "route" in expect and (state.route.value if state.route else None) != expect["route"]:
        reasons.append(f"route: expected {expect['route']!r}, got {state.route}")

    if "tool_used" in expect and state.selected_tool != expect["tool_used"]:
        reasons.append(f"tool_used: expected {expect['tool_used']!r}, got {state.selected_tool!r}")

    if "error_code" in expect:
        actual = state.error_code.value if state.error_code else None
        if actual != expect["error_code"]:
            reasons.append(f"error_code: expected {expect['error_code']!r}, got {actual!r}")

    if "refused" in expect and state.refused != expect["refused"]:
        reasons.append(f"refused: expected {expect['refused']}, got {state.refused}")

    if "citations" in expect and expect["citations"] == [] and state.retrieved_sources:
        reasons.append(f"citations: expected none, got {[c.chunk_id for c in state.retrieved_sources]}")

    if "min_citations" in expect and len(state.retrieved_sources) < expect["min_citations"]:
        reasons.append(f"min_citations: expected >= {expect['min_citations']}, got {len(state.retrieved_sources)}")

    if "answer_contains" in expect:
        missing = _contains_all(state.answer or "", expect["answer_contains"])
        if missing:
            reasons.append(f"answer_contains: missing {missing}")

    if "agent_path_contains" in expect:
        path = [step.agent for step in state.plan]
        missing = [a for a in expect["agent_path_contains"] if a not in path]
        if missing:
            reasons.append(f"agent_path_contains: missing {missing} (got {path})")

    if expect.get("no_write_triggered"):
        if state.next_action == NextAction.AWAIT_APPROVAL or state.approval_required:
            reasons.append("no_write_triggered: an embedded document instruction triggered a pending write")

    if "approval_required" in expect and state.approval_required != expect["approval_required"]:
        reasons.append(f"approval_required: expected {expect['approval_required']}, got {state.approval_required}")

    return CaseResult(
        id=case["id"], family=case["family"], passed=not reasons,
        acceptance_rule=case["acceptance_rule"], reasons=reasons,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def check_retry(case: dict, harness: Harness) -> CaseResult:
    expect = case["expect"]
    reasons: list[str] = []
    started = time.perf_counter()

    registry = _flaky_registry(harness.tool_registry, "get_project_status", case["fail_times"])
    state = harness.run_chat(case["prompt"], case.get("role"), tool_registry=registry)

    actual_error = state.error_code.value if state.error_code else None
    if "error_code" in expect and actual_error != expect["error_code"]:
        reasons.append(f"error_code: expected {expect['error_code']!r}, got {actual_error!r}")

    result = state.agent_results.get("erp_analyst")
    retries = result.retries if result else 0
    if "min_retry_count" in expect and retries < expect["min_retry_count"]:
        reasons.append(f"min_retry_count: expected >= {expect['min_retry_count']}, got {retries}")

    return CaseResult(
        id=case["id"], family=case["family"], passed=not reasons,
        acceptance_rule=case["acceptance_rule"], reasons=reasons,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def check_approval(case: dict, harness: Harness) -> CaseResult:
    expect = case["expect"]
    reasons: list[str] = []
    started = time.perf_counter()

    pending = harness.run_chat(case["prompt"], case.get("role"))

    if expect.get("pending_first") and pending.next_action != NextAction.AWAIT_APPROVAL:
        reasons.append(f"pending_first: expected AWAIT_APPROVAL, got {pending.next_action}")
        return CaseResult(
            case["id"], case["family"], False, case["acceptance_rule"], reasons,
            (time.perf_counter() - started) * 1000,
        )

    approved = case["decision"] == "approve"
    tool_input = {**(pending.tool_input or {})}
    if approved and "risk_payload" in case:
        tool_input["risk_payload"] = case["risk_payload"]
    decided = pending.model_copy(update={"approved": approved, "tool_input": tool_input})

    resumed = resume_multi_agent(
        decided, tool_registry=harness.tool_registry, agents=harness.agents,
        authorize=check_permission,
    )

    actual_error = resumed.error_code.value if resumed.error_code else None
    if "error_code" in expect and actual_error != expect["error_code"]:
        reasons.append(f"error_code: expected {expect['error_code']!r}, got {actual_error!r}")

    if "answer_contains" in expect:
        missing = _contains_all(resumed.answer or "", expect["answer_contains"])
        if missing:
            reasons.append(f"answer_contains: missing {missing}")

    return CaseResult(
        id=case["id"], family=case["family"], passed=not reasons,
        acceptance_rule=case["acceptance_rule"], reasons=reasons,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def check_memory_decision(case: dict, harness: Harness) -> CaseResult:
    expect = case["expect"]
    reasons: list[str] = []
    started = time.perf_counter()

    memory = LongTermMemory()
    if "seed_text" in case:
        memory = memory.add(MemoryEntry(
            memory_id="SEED-1", text=case["seed_text"], written_at=time.time(),
            source="agent_inference", subject=case["candidate"].get("subject", ""),
        ))

    candidate = MemoryCandidate(**case["candidate"])
    decision = decide(candidate, memory)

    if "action" in expect and decision.action.value != expect["action"]:
        reasons.append(f"action: expected {expect['action']!r}, got {decision.action.value!r}")

    if "reason_contains" in expect:
        missing = _contains_all(decision.reason, expect["reason_contains"])
        if missing:
            reasons.append(f"reason_contains: missing {missing} (reason: {decision.reason!r})")

    return CaseResult(
        id=case["id"], family=case["family"], passed=not reasons,
        acceptance_rule=case["acceptance_rule"], reasons=reasons,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


_CHECKS = {
    "chat": check_chat,
    "retry": check_retry,
    "approval": check_approval,
    "memory_decision": check_memory_decision,
}


def run_dataset(dataset_path: Path = DATASET_PATH) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    harness = Harness()

    results: list[CaseResult] = []
    for case in dataset["cases"]:
        checker = _CHECKS[case["kind"]]
        try:
            results.append(checker(case, harness))
        except Exception as exc:  # noqa: BLE001 - a crashing case is a failure, not a runner crash
            results.append(CaseResult(
                id=case["id"], family=case.get("family", "unknown"), passed=False,
                acceptance_rule=case.get("acceptance_rule", ""),
                reasons=[f"case raised {type(exc).__name__}: {exc}"],
            ))

    critical_ids = set(dataset.get("critical_case_ids", []))
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    critical_results = [r for r in results if r.id in critical_ids]
    critical_passed = sum(1 for r in critical_results if r.passed)

    overall_rate = passed / total if total else 0.0
    critical_rate = critical_passed / len(critical_results) if critical_results else 1.0

    report = {
        "dataset_version": dataset["version"],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "overall_pass_rate": round(overall_rate, 4),
        "overall_threshold": OVERALL_PASS_THRESHOLD,
        "overall_gate_passed": overall_rate >= OVERALL_PASS_THRESHOLD,
        "critical_case_ids": sorted(critical_ids),
        "critical_pass_rate": round(critical_rate, 4),
        "critical_threshold": CRITICAL_PASS_THRESHOLD,
        "critical_gate_passed": critical_rate >= CRITICAL_PASS_THRESHOLD,
        "cases": [
            {
                "id": r.id, "family": r.family, "passed": r.passed,
                "acceptance_rule": r.acceptance_rule, "reasons": r.reasons,
                "elapsed_ms": round(r.elapsed_ms, 1), "critical": r.id in critical_ids,
            }
            for r in results
        ],
    }
    report["gate_passed"] = report["overall_gate_passed"] and report["critical_gate_passed"]
    return report


def _write_report(report: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"report-{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = REPORTS_DIR / "latest.json"
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _print_report(report: dict) -> None:
    print(f"dataset v{report['dataset_version']} — {report['total_cases']} cases")
    print(
        f"overall:  {report['passed']}/{report['total_cases']} "
        f"({report['overall_pass_rate']:.0%}, threshold {report['overall_threshold']:.0%}) "
        f"{'PASS' if report['overall_gate_passed'] else 'FAIL'}"
    )
    print(
        f"critical: {report['critical_pass_rate']:.0%} of {len(report['critical_case_ids'])} "
        f"(threshold {report['critical_threshold']:.0%}) "
        f"{'PASS' if report['critical_gate_passed'] else 'FAIL'}"
    )
    print()
    for case in report["cases"]:
        marker = "PASS" if case["passed"] else "FAIL"
        crit = " [critical]" if case["critical"] else ""
        print(f"  {marker}  {case['id']:<16} {case['family']:<20}{crit}")
        for reason in case["reasons"]:
            print(f"        - {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.runner", description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = run_dataset(Path(args.dataset))
    path = _write_report(report)

    if not args.quiet:
        _print_report(report)
        print(f"\nreport written to {path}")

    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
