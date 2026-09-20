# Deployment Policy

Last updated: internal engineering wiki, Meridian Systems platform team.

## Overview
All production deployments for Meridian Systems services must go through the
staged rollout pipeline. This applies to every service in the `core-platform`
and `billing` repositories. Services in the `sandbox` repository are exempt.

## Deployment windows
* Standard deployments: Tuesday–Thursday, 10:00–16:00 local time.
* No deployments on Fridays, weekends, or the last business day of the fiscal quarter.
- Emergency hotfixes are exempt from the deployment window restriction but still require sign-off (see below).

### Sign-off requirements

| Change type            | Required approvals            | Rollback plan required? |
|-------------------------|-------------------------------|--------------------------|
| Minor (config/feature flag) | 1 engineer                | No                       |
| Standard (code change)  | 1 engineer + 1 tech lead      | Yes                      |
| Major (schema/infra)    | Tech lead + on-call SRE       | Yes, tested              |
| Emergency hotfix        | On-call SRE (post-hoc review) | Yes                      |

## Rollout stages
1. Canary — 5% of traffic, minimum 30 minutes observation.
2. Partial — 25% of traffic, minimum 1 hour observation.
3. Full — 100% of traffic.

Each stage requires the error rate to stay below 0.5% above baseline and p95
latency to stay within 20% of baseline. If either threshold is breached, the
rollout auto-rolls-back to the previous version.

#### Feature flags
Feature flags should be used for any user-facing behavior change so it can be
disabled without a full redeploy. Flags older than 90 days should be cleaned
up during the next sprint touching that service.

## Related documents
See the Incident Response Runbook for what happens if a deployment causes a
production incident, and the On-Call Rotation guide for who to page if an
emergency hotfix is needed outside business hours.
