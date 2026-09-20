# Testing Standards

## Test pyramid targets
Meridian Systems services aim for roughly:
* 70% unit tests — fast, no network or database, run on every commit.
* 20% integration tests — real database (test container), mocked
  third-party APIs, run on every PR.
* 10% end-to-end tests — full stack against a staging environment, run
  before every deployment (see Deployment Policy).

## Coverage requirements
New code requires at least 80% line coverage. Coverage is measured on the
diff, not the whole file, so touching one function in a large legacy file
doesn't require testing the whole file. Coverage is enforced in the merge
queue described in the Code Review Guidelines.

## Flaky tests
A test that fails intermittently without a code change is quarantined
(marked `@flaky`, excluded from blocking CI) within 24 hours of being
identified, with a tracking ticket. Quarantined tests must be fixed or
deleted within 2 weeks — they cannot stay quarantined indefinitely.

## What must have tests
- Every new API endpoint needs at least: one happy-path test, one
  validation-error test, and one auth-failure test.
- Every bug fix needs a regression test that fails without the fix.
- Database migrations need a test that runs the expand phase against a
  copy of production-shaped data (see Database Migration Guide).

## Load testing
Any change expected to increase traffic to a service by more than 20%
(new feature launch, marketing campaign, etc.) requires a load test against
staging at 1.5x the expected peak load, signed off by the on-call SRE for
that service, at least 3 business days before launch.

## Test data
Synthetic test data only — production data, including anonymized or
sampled production data, must never be copied into test or staging
environments. This applies especially to anything classified as
Confidential or Restricted under the Security Policy.
