---
doc_id: DOC-SLA-001
title: ERP Vendor Service Level Agreement (Extract)
project_code: ALL
classification: public
owner: Vendor Management
updated_at: 2026-03-18
---

# ERP Vendor Service Level Agreement (Extract)

## Availability

The vendor commits to 99.5% monthly availability for the production tenant,
measured excluding scheduled maintenance windows. Scheduled maintenance is
capped at four hours per month and must be announced ten business days ahead.

## API rate limits

The bulk synchronisation endpoint is throttled at 500 requests per minute per
tenant. Requests above the limit receive a 429 response with a Retry-After
header. Sustained breach of the limit for more than five minutes may result in
temporary suspension of the integration credential.

Clients are expected to implement bounded retry with exponential backoff.
Retrying immediately on a 429, or retrying without a jitter, is explicitly
called out in the agreement as client misuse and voids the availability credit
for the affected period.

## Support response targets

- Severity 1 (production down): 1 hour response, 4 hour workaround target.
- Severity 2 (major function degraded): 4 hour response, 2 business day target.
- Severity 3 (minor or cosmetic): 2 business day response, next release.

## Data handling

The vendor processes data as a processor, not a controller. Customer data is
stored in the contracted region and is not replicated outside it. API keys are
issued per integration user and must be rotated at least every 12 months.

## Credit and remedies

Failure to meet the availability commitment entitles the customer to a service
credit of 5% of the monthly fee per 0.5% below target, capped at 25%. Credits
must be claimed within 30 days of the affected month.

## Termination assistance

On termination, the vendor provides a full data export in a documented format
within 30 days, and read-only access for a further 60 days.
