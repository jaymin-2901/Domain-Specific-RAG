# On-Call Rotation Guide

## Schedule
On-call rotates weekly, Monday 10:00 to the following Monday 10:00 local
time. Each service team maintains its own primary and secondary rotation;
the platform team's rotation additionally covers shared infrastructure.

## Escalation tiers
1. Primary on-call — first page for any severity.
2. Secondary on-call — paged automatically if the primary doesn't
   acknowledge within the incident's SLA window (5 minutes for P1, per the
   Incident Response Runbook).
3. Engineering manager — paged if secondary also doesn't acknowledge within
   a further 5 minutes.
4. Director of Engineering — paged only for unresolved P1 incidents
   exceeding 2 hours.

## Compensation
Engineers on primary on-call receive on-call pay for the full week
regardless of whether they are paged, plus overtime pay for any hours
actively spent resolving an incident outside business hours.

## Handoff checklist
- [ ] Review open incidents and their current status
- [ ] Review any scheduled maintenance for the upcoming week
- [ ] Confirm paging app notifications are working on your phone
- [ ] Read the previous on-call's handoff notes in the on-call channel

## Swapping shifts
Shift swaps must be requested at least 48 hours in advance and confirmed by
both engineers in the on-call channel. Swaps within 48 hours require tech
lead approval. There is no swap process for an already-active on-call shift
except in a documented emergency (e.g., illness), which requires manager
notification.

## Tooling
The paging system is configured per-service; engineers should verify their
contact methods (phone, app, email) are current in the paging tool before
their rotation begins. Escalation timing is enforced automatically by the
paging tool and mirrors the tiers above.
