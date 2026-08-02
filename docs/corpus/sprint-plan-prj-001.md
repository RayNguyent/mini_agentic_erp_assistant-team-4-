---
doc_id: DOC-SPRINT-001
title: PRJ-001 Sprint Plan and Delivery Cadence
project_code: PRJ-001
classification: internal
owner: Alice Tran
updated_at: 2026-07-20
---

# PRJ-001 Sprint Plan and Delivery Cadence

## Cadence

Sprints are two weeks long, starting on a Monday. Planning happens on day one,
a mid-sprint checkpoint on day five, and review plus retrospective on day ten.
The team commits to a scope at planning and does not accept mid-sprint additions
unless an equivalent amount of work is removed.

## Definition of done

A task is done when the code is merged, unit tests cover the new branch,
the migration script has been rehearsed against a production-shaped dataset,
and the acceptance criteria have been confirmed by the requesting business
owner. Work that is merged but not rehearsed is counted as open, not done.

## Iteration mapping

The delivery iteration is tracked as a milestone. Every task carries a milestone
reference; tasks without one are backlog items and are excluded from sprint
progress calculations. This mapping matters because the underlying ERP does not
provide a native sprint object — the milestone is the sprint.

## Sprint 4 (current)

Sprint 4 runs 2026-07-13 to 2026-07-24 and covers purchase-to-pay plus vendor
onboarding. The team committed to 12 tasks. As of the mid-sprint checkpoint,
7 were complete, 4 were in progress, and 1 was blocked on vendor API access.

## Overdue policy

A task is overdue when its deadline has passed and its state is neither done
nor cancelled. Overdue tasks are reviewed at the daily checkpoint and either
re-estimated or explicitly deferred to the next milestone. Silently carrying an
overdue task forward without a decision is treated as a process defect.

## Velocity

Measured velocity across sprints 1-3 was 11, 14, and 12 completed tasks. The
team plans against the trailing three-sprint average rather than the best
observed sprint, which is why sprint 4 committed to 12 rather than 14.

## Escalation

If a sprint finishes below 70% of committed scope twice in a row, the project
manager raises it to the steering committee with a re-baselining proposal.
