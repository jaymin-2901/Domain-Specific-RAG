Security Policy
===============

### Scope
This document covers application-level security requirements for services
in `core-platform`, `billing`, and any service handling customer PII.

## Secrets management
* Secrets must never be committed to a repository, including in test
  fixtures or comments.
* All secrets are stored in the central vault and injected at runtime as
  environment variables.
* Rotate all production secrets at least every 180 days, or immediately
  after any suspected exposure.

## Authentication & authorization
- Every internal service-to-service call must use mutual TLS.
- Customer-facing endpoints must use OAuth 2.0 with short-lived access
  tokens (max 1 hour) and refresh tokens capped at 30 days.
- Role checks must happen server-side; never trust a role claim without
  verifying it against the authorization service on every request.

## Data handling

| Data class      | Encryption at rest | Encryption in transit | Retention        |
|------------------|--------------------|-----------------------|--------------------|
| Public            | Not required        | TLS 1.2+               | Indefinite          |
| Internal          | Required             | TLS 1.2+               | Indefinite          |
| Confidential (PII)| Required (AES-256)  | TLS 1.3                | 7 years, then purge |
| Restricted (payment) | Required (AES-256) + tokenization | TLS 1.3 | Per PCI-DSS requirements |

## Vulnerability response
Security findings are triaged using the same severity levels as the
Incident Response Runbook (P1–P4), with one exception: any Restricted-data
exposure is automatically treated as P1 regardless of apparent scope, until
the security team confirms otherwise.

## Prompt injection & AI systems
Any internal tool that feeds untrusted text (documents, user messages,
scraped content) into an LLM prompt must treat that text as data, never as
instructions, and should log any detected attempt to override system
instructions for security review. This applies to both customer-facing
assistants and internal tooling such as the engineering-docs search system.

## Code review requirement
Per the Code Review Guidelines, any change touching authentication,
authorization, secrets handling, or the payments service requires two
approvals, at least one from a member of the security team.
