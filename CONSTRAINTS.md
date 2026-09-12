# Project Quality Constraints & Anti-Drift Contract

> Enforced by `skills/constraint-driven-development` and `skills/using-agent-skills`.
> This document defines the non-negotiable quality floor and anti-drift boundaries for this repository.

## 1. Zero-Drift Policy (Mandatory Skill Usage)

Before executing any task, the agent MUST explicitly identify and apply the corresponding skill:

| Activity / Task | Mandatory Skill | Enforcement Rule |
|---|---|---|
| **Architecture choices, trade-offs, forks** | `documentation-and-adrs` | An ADR (`ADR-xxx`) MUST be created in `docs/decisions/` before code lock-in. |
| **Requirements exploration & ideation** | `idea-refine` / `interview-me` | Clarify constraints and surface assumptions before writing code. |
| **New module or API contract design** | `api-and-interface-design` + `spec-driven-development` | Define Pydantic models / Protocols in `core/interfaces.py` before concrete implementations. |
| **Task execution & chunking work** | `planning-and-task-breakdown` + `incremental-implementation` | Break into thin, verifiable slices (<3 files modified per slice). |
| **All code additions and bug fixes** | `test-driven-development` | Red-Green-Refactor. Test written and verified passing before committing. |
| **Code evaluation & cleanup** | `code-review-and-quality` + `code-simplification` | Check 5 axes (correctness, readability, architecture, security, performance). |
| **Security, input sanitization, guardrails** | `security-and-hardening` | Check OWASP Top 10, untrusted document sandboxing, prompt injection barriers. |
| **Git commits & versions** | `git-workflow-and-versioning` | Conventional commits (`feat(...)`, `docs(adr)`, `fix(...)`, `test(...)`). |

## 2. Anti-Drift Guardrails

1. **Phase Discipline**: Work MUST follow the designated roadmap phases (Phase 1 Ingestion -> Phase 2 Storage -> Phase 3 Hybrid Retrieval -> Phase 4 Reranking -> Phase 5 Synthesis -> Phase 6 API & UI -> Phase 7 Packaging). Downstream components (e.g., FastAPI endpoints) MUST NOT be prototyped until upstream components (Qdrant storage & hybrid search) are tested and green.
2. **Dependency Integrity**: No unvetted, bloated dependencies. Every new dependency must be added to `pyproject.toml` with an explicit reason recorded.
3. **No Silencing Checks**:
   - Zero test skipping (`@pytest.mark.skip` forbidden without written justification).
   - Zero silenced exceptions with bare `except: pass`.
   - Test assertions must test behavior, not trivial identity (`assert True` is forbidden).
4. **Environment Integrity**:
   - All tests MUST be executed against the virtual environment (`.venv/bin/python -m pytest`).
   - 100% test pass rate required before any commit.
5. **No Hallucinated Implementations**:
   - No stubbed functions left as `pass` or `raise NotImplementedError` in committed code unless explicitly part of an abstract protocol.
