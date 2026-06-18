<p align="center">
  <strong>dragonlight-router</strong>
</p>
<p align="center">
  <em>Multi-provider LLM routing engine — intelligent model selection and cascade dispatch.</em>
</p>
<p align="center">
  <a href="https://github.com/dragonlightintl/dragonlight-router/actions/workflows/ci.yml"><img src="https://github.com/dragonlightintl/dragonlight-router/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/dragonlightintl/dragonlight-router"><img src="https://codecov.io/gh/dragonlightintl/dragonlight-router/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://dragonlightintl.github.io/dragonlight-router/"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
  <a href="https://pypi.org/project/dragonlight-router/"><img src="https://img.shields.io/pypi/v/dragonlight-router.svg" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="llms.txt"><img src="https://img.shields.io/badge/llms.txt-available-blue.svg" alt="llms.txt"></a>
</p>
<p align="center">
  <b>Documentation</b>: <a href="https://dragonlightintl.github.io/dragonlight-router/">dragonlightintl.github.io/dragonlight-router</a>
</p>

---

**One component replaces the provider if/else, the health-check spreadsheet, and the rate-limit math.**

dragonlight-router is a multi-provider LLM routing engine. It selects the best available model for each request across 11 providers, dispatches through a three-stage cascade, and handles fallback automatically when providers fail.

## Features

- **11 providers** — Anthropic, Cerebras, Cohere, Gemini, Groq, Mistral, NVIDIA NIM, Ollama, OpenAI, OpenRouter, Together
- **MBR → CBR → LBR cascade** — capability filtering, cost-aware scoring, rate-limit enforcement in three independent stages
- **Circuit breaking** — failing models are removed from rotation until they recover
- **Budget enforcement** — sliding-window RPM, RPD, and TPM tracking per provider
- **Dual interface** — get a ranked model list (you call the LLM) or hand off the entire request (router dispatches with fallback)
- **Hot-reload** — change the role-to-model matrix or provider config without restarting
- **Response caching** — exact-match and semantic near-duplicate caching backed by SQLite
- **SSE streaming** — `dispatch_stream()` yields token-by-token events with automatic fallback

## Quickstart

```bash
pip install -e ".[all]"
```

### Python library

```python
from dragonlight_router import RouterEngine
from dragonlight_router.core.types import RequestOutcome

router = RouterEngine()  # loads config/router.yaml

# Ranked models for a role — your app makes the LLM call
models = router.select_models("code_review", top_n=5)
# → ["groq/llama-3.3-70b-versatile", "cerebras/llama3.1-70b", ...]

# Record the outcome so budget + health stay accurate
router.record_request(RequestOutcome(
    provider="groq", model_id=models[0],
    success=True, tokens_used=1024, latency_ms=340.0,
))
```

### HTTP sidecar

```bash
dragonlight-router                    # starts on http://127.0.0.1:8100

curl -s -X POST http://127.0.0.1:8100/v1/select \
  -H "Content-Type: application/json" \
  -d '{"role": "summarize", "top_n": 3}' | jq .
```

## Documentation

| Resource | Link |
|---|---|
| Full docs | [dragonlightintl.github.io/dragonlight-router](https://dragonlightintl.github.io/dragonlight-router/) |
| Architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| API reference | [docs/openapi.yaml](docs/openapi.yaml) |
| Configuration | [docs/live-specs/](docs/live-specs/) |
| Deployment | [docs/deployment-runbook.md](docs/deployment-runbook.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

## Cascade pipeline

```
Request → MBR (filter by capability + tier + status)
        → CBR (score on budget headroom + health)
        → LBR (enforce rate limits, interleave providers, weighted random)
        → Adapter dispatch (call provider, walk fallback chain on failure)
        → Response
```

## Running tests

```bash
pip install -e ".[all,dev]"
make test              # fast: no coverage
make test-cov          # with coverage report
make all               # lint + typecheck + test
```

## License

MIT. See [LICENSE](LICENSE).
