---
doc_id: DOC-CHARTER-001
title: PRJ-001 ERP Platform Rollout — Project Charter
project_code: PRJ-001
classification: internal
owner: Alice Tran
updated_at: 2026-05-12
---

# PRJ-001 ERP Platform Rollout — Project Charter

## Purpose

PRJ-001 replaces the legacy on-premise finance and inventory stack with a single
hosted ERP platform. The rollout consolidates four regional instances into one
production tenant and retires the nightly CSV reconciliation job that currently
causes a 14-hour reporting lag.

## Scope

In scope: general ledger migration, inventory master data, purchase-to-pay,
vendor onboarding, and the reporting warehouse feed.

Out of scope: payroll, the customer-facing storefront, and the Warehouse
Automation robotics integration, which is tracked separately as PRJ-002.

## Success criteria

The rollout is considered successful when all four regional instances are
decommissioned, month-end close completes within 3 business days (down from 9),
and no more than 2 severity-high defects remain open at go-live.

## Governance

Alice Tran is the accountable project manager. The steering committee meets
fortnightly. Any change that moves the go-live date or increases the approved
budget by more than 5% requires steering committee sign-off; the project manager
cannot approve those alone.

## Delivery approach

Delivery runs in six two-week sprints. Sprints 1-3 covered data migration and
the finance core. Sprint 4, currently in progress, covers purchase-to-pay and
vendor onboarding. Sprints 5 and 6 cover reporting, cutover rehearsal, and
hypercare.

## Key dates

Discovery closed 2026-02-10. Build started 2026-03-02. User acceptance testing
opens 2026-08-17. Go-live is planned for 2026-09-28 with a two-week hypercare
window following.

## Dependencies

The rollout depends on the vendor's bulk sync API (see the vendor SLA document)
and on the finance module migration in PRJ-003 completing its chart-of-accounts
mapping before UAT opens.
