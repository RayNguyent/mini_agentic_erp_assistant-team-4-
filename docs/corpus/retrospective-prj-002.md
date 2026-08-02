---
doc_id: DOC-RETRO-002
title: PRJ-002 Warehouse Automation — Discovery Retrospective
project_code: PRJ-002
classification: internal
owner: Minh Pham
updated_at: 2026-06-28
---

# PRJ-002 Warehouse Automation — Discovery Retrospective

## Context

PRJ-002 completed its discovery phase in June 2026. The project is currently in
Discovery stage and has not yet entered build. This retrospective covers the
discovery phase only.

## What went well

Site surveys at all three distribution centres finished ahead of schedule. The
robotics vendor shortlist was reduced from nine to three candidates with a clear
scoring rubric, which made the selection defensible to procurement. Stakeholder
engagement from warehouse operations was consistently strong.

## What did not go well

Integration assumptions with PRJ-001 were made verbally and never written down,
which produced two weeks of rework when the inventory master data model changed.
Estimates for the conveyor control interface were made without access to the
existing PLC documentation and are considered low-confidence.

## Actions

Integration contracts with PRJ-001 must be written and reviewed before build
starts. The conveyor interface estimate is to be re-done after a two-week
technical spike. A named integration owner is to be appointed across both
projects.

## Risk register state

PRJ-002 currently has no risks recorded in the register. This is a finding, not
a good outcome: a project entering build with an empty register almost certainly
has unrecorded risk rather than no risk. The project manager was actioned to run
a risk identification workshop before the build gate.

## Readiness for build

The discovery gate was passed with conditions. Build cannot start until the
integration contract and the risk workshop are complete.
