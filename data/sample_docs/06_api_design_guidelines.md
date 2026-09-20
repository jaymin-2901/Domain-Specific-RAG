# API Design Guidelines

## General principles
APIs should be resource-oriented, versioned in the URL path (`/v1/...`), and
return consistent error shapes across all services.

## Error format
All errors return the following JSON shape:

    {
      "error": {
        "code": "RESOURCE_NOT_FOUND",
        "message": "human readable message",
        "request_id": "uuid"
      }
    }

Error codes are UPPER_SNAKE_CASE and documented per-endpoint. HTTP status
codes should match the error semantics (404 for not found, 409 for
conflict, 422 for validation errors, 429 for rate limiting, 5xx only for
genuine server faults).

## Pagination
* Use cursor-based pagination (`?cursor=...&limit=...`), not offset-based,
  for any collection that can grow past 1,000 items.
* Default `limit` is 50, max is 200.
* Responses include a `next_cursor` field, `null` when there are no more
  pages.

## Rate limits
Public API rate limits are tiered by plan (see the billing service docs for
exact numbers) and returned in `X-RateLimit-*` response headers on every
request, not just when a limit is hit.

## Idempotency
Any endpoint that creates a resource (POST) must accept an
`Idempotency-Key` header and guarantee that retried requests with the same
key do not create duplicate resources, for at least 24 hours after the
original request.

## Deprecation policy
- A deprecated field or endpoint must remain functional for at least 6
  months after the deprecation is announced.
- Deprecation notices are sent via the developer changelog and an HTTP
  `Deprecation` response header.
- Breaking changes always require a new API version; they are never made
  in place on an existing version.

## Backward compatibility for schema changes
See the Database Migration Guide for how underlying schema changes should
be sequenced so that API responses never break mid-rollout.
