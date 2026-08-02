---
doc_id: DOC-ADR-003
title: PRJ-003 Finance Module Migration — Architecture Decision Log
project_code: PRJ-003
classification: internal
owner: Bao Nguyen
updated_at: 2026-07-08
---

# PRJ-003 Finance Module Migration — Architecture Decision Log

## AD-1: Chart of accounts is remapped, not lifted

The legacy chart of accounts carries 1,400 accounts, roughly 400 of which have
had no posting in three years. We remap to a rationalised 900-account structure
rather than lifting the legacy structure as-is.

Rejected alternative: a straight lift-and-shift. It would have been faster to
migrate but would have carried the reconciliation problem into the new platform
and blocked the reporting simplification that justified the project.

Consequence: PRJ-001 cannot open user acceptance testing until this mapping is
signed off, because the reporting warehouse feed depends on the new structure.

## AD-2: Historical data is archived, not migrated

Only two full fiscal years of transactional history move into the new platform.
Older history is archived to a read-only store with a documented query path.

Rejected alternative: migrating seven years of history. It would have tripled
the migration window and the archive satisfies the statutory retention
requirement equally well.

## AD-3: Integration identifiers are explicit, never inferred

Cross-system records are matched on an explicit external identifier stored on
the record, never on the display name. Name matching was used in a prior
programme and produced silent mismatches for entities with similar names.

Rejected alternative: fuzzy name matching with a confidence threshold. Rejected
because a silent wrong match in finance data is materially worse than a failed
lookup that surfaces an error.

## AD-4: Reconciliation runs before cutover, not after

A full reconciliation pass runs against a production-shaped dataset before
cutover, and the cutover is blocked if the pass reports any unmatched balance.

## Current status

PRJ-003 is in user acceptance testing. AD-1 mapping sign-off is the critical
path item for PRJ-001's UAT gate.
