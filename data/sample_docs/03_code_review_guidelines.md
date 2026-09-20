Code Review Guidelines
======================

These guidelines apply to all pull requests merged into `main` across
Meridian Systems repositories.

## Before requesting review
* Run the full test suite locally, not just the changed module's tests.
* Keep PRs under ~400 changed lines where possible; split larger changes
  into a stacked series of PRs.
* Write a description that explains *why*, not just *what* — the diff
  already shows what changed.

## Reviewer responsibilities

1. Respond to review requests within 1 business day (P1/P2 incident fixes
   are prioritized ahead of normal review queues — see Incident Response).
2. At least one approval is required to merge; two are required for changes
   touching authentication, billing, or the payments service.
3. Reviewers should distinguish blocking comments from suggestions. Prefix
   non-blocking comments with "nit:" or "suggestion:".

## What to look for
- Correctness and edge cases (empty inputs, concurrent access, timezones)
- Test coverage for new logic, especially error paths
- Whether the change needs a feature flag (see Deployment Policy)
- Security implications — anything touching user input, auth, or secrets
  should also satisfy the checklist in the Security Policy document
- Readability: would a new team member understand this in six months?

## Merge queue
PRs are merged via the automated merge queue, which re-runs CI against the
latest `main` before merging to prevent semantic conflicts. Do not use
"merge without waiting for checks" outside of a declared P1 incident.

## Style
Meridian Systems uses automated formatters (see `.editorconfig` in each
repo) — style nitpicks that the formatter would catch should not be raised
in review. Focus review time on logic, architecture, and correctness.
