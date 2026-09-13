# ADR-014: UI Model Allowlist and Per-Request Synthesis Override

## Status
Accepted

## Date
2026-09-13

## Context
Recall's embedded Web UI (ADR-010) displays the active synthesis model and sends chat requests to `POST /v1/chat`. Operators configure the default LLM via `.env` (`LOCAL_LLM_MODEL`) and OpenRouter, but:

1. Changing models requires a server restart.
2. End users cannot compare models during demos without editing server config.
3. Exposing an unrestricted model string would allow arbitrary OpenRouter spend.

## Decision

1. **Allowlist in `config.yaml`** under `ui.models` — each entry has `id` (OpenRouter slug, e.g. `openai/gpt-oss-120b`) and `label` (display name).
2. **`GET /v1/config`** returns the allowlist, `default_model`, `active_model` (server startup default), and pipeline defaults for the settings panel.
3. **`POST /v1/chat`** accepts optional `model` — validated against the allowlist; builds a per-request `Synthesizer` instance (thread-safe).
4. **UI** stores the user's selection in `localStorage` and sends `model` on each chat request.
5. If `ui.models` is empty, the server exposes only the resolved default model (backward compatible).

## Alternatives Considered

- **Mutate shared `Synthesizer` on each request** — rejected; not safe under concurrent requests.
- **Free-text model input in UI** — rejected; cost and compliance risk.
- **Server-persisted user preferences** — deferred; requires auth and storage.

## Consequences

- Admins control available models via version-controlled `config.yaml`.
- Users switch models without restart; badge stays in sync via `/v1/config` refresh.
- Per-request synthesizer construction is negligible vs LLM latency (~5s+).
