Database Migration Guide
=========================

## Why phased migrations
Schema changes must never require the application and database to deploy
in lockstep, because deployments are gradual (see the Deployment Policy's
canary/partial/full stages) — for part of every rollout, old and new code
are running against the same database simultaneously.

## The expand/contract pattern
1. **Expand** — add the new column/table/index without removing or
   renaming anything old. Deploy application code that can write to both
   old and new schema (dual-write) but still reads from the old schema.
2. **Migrate** — backfill historical data into the new schema. Deploy
   application code that reads from the new schema, falling back to old
   if missing.
3. **Contract** — once 100% of traffic is confirmed reading/writing the
   new schema successfully (minimum one full deployment cycle with no
   rollbacks), remove the old column/table and the dual-write code.

Never skip straight to contract. A rollback during the expand or migrate
phase is safe (old code still works); a rollback after contract is not.

## Backfills
* Backfills run as a background job, batched (default 1,000 rows per
  batch), with a configurable sleep between batches to avoid saturating
  database IOPS.
* Large backfills (>10M rows) require sign-off from the on-call SRE and
  should be scheduled outside the standard deployment window if they could
  affect production query latency.

## Index changes
Adding an index on a large table should always use
`CREATE INDEX CONCURRENTLY` (Postgres) to avoid locking writes. Dropping an
unused index requires confirming zero query plan usage over the prior
30 days first.

## Rollback
If a migration causes a P1 or P2 incident, follow the Incident Response
Runbook. Migrations themselves are not rolled back mid-flight; instead,
the *application* is rolled back to the previous expand-phase version,
which is guaranteed to be compatible with both schema versions.
