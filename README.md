# Enterprise Production RAG Kit

A modular, plug-and-play Retrieval-Augmented Generation (RAG) framework engineered for enterprise scale (10M+ documents).

## Architecture & Standards
- Architecture Decision Records (ADRs) are documented in [`docs/decisions/`](docs/decisions/).
- Follows [`AGENTS.md`](AGENTS.md) engineering discipline powered by [`addyosmani/agent-skills`](.agents/).

## Development
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,pdf]"
pytest
```
