# Incident Response Runbook

## Severity levels

- P1 — Full outage or data loss affecting all customers. Page immediately.
- P2 — Partial outage or major feature broken for a subset of customers.
- P3 — Degraded performance or minor feature broken, workaround exists.
- P4 — Cosmetic issue, no customer impact.

## Response SLAs

| Severity | Acknowledgement SLA | Update cadence | Resolution target |
|----------|---------------------|-----------------|--------------------|
| P1       | 5 minutes            | every 15 minutes | 4 hours            |
| P2       | 15 minutes           | every 30 minutes | 24 hours           |
| P3       | 4 hours              | daily            | 5 business days    |
| P4       | Next business day    | N/A              | Best effort        |

## Escalation

If a P1 incident is not acknowledged within its 5 minute SLA, the paging
system automatically escalates to the secondary on-call engineer, and after
a further 5 minutes, to the on-call engineering manager. This escalation
chain is the same one described in the On-Call Rotation guide's "escalation
tiers" section — the incident tooling and the on-call schedule are kept in
sync automatically.

### Declaring an incident
1. Page via the incident bot (`/incident declare`) with a one-line summary.
2. A dedicated incident channel is created automatically.
3. Assign an Incident Commander (IC) — usually the on-call engineer who
   acknowledged the page, unless a more senior engineer is already engaged.
4. The IC owns communication and delegates investigation; they do not have
   to personally debug the issue.

## Postmortems
Every P1 and P2 incident requires a blameless postmortem within 5 business
days of resolution. Postmortems must include a timeline, root cause, and at
least one action item with an owner and due date. Postmortems are reviewed
in the weekly platform sync.

## Rollback during an incident
If a recent deployment is suspected as the cause, follow the rollback
procedure in the Deployment Policy document rather than attempting a forward
fix, unless the IC explicitly decides a forward fix is faster and lower risk.
