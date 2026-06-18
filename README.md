# dragonlight-router

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

Every non-trivial LLM application accumulates the same debt: a growing if/else over provider names, hand-rolled rate counters, a health-check spreadsheet nobody updates, and a deployment that sends all traffic to one provider until it falls over.

**dragonlight-router** replaces that pile with a single component.

---

## What you get

- **Intelligent model selection** across 11 providers — one call returns a ranked list tuned to your task
- **Three-stage cascade dispatch** (capability filtering, cost scoring, load balancing) so every request hits the best available model
- **Automatic health tracking and circuit breaking** — failing models are removed from rotation, not retried until they recover
- **Budget enforcement** — sliding-window RPM, RPD, and TPM tracking per provider with configurable limits
- **Dual interface** — get a ranked model list (you call the LLM) or hand off the entire request (router calls it, handles fallback)
- **Hot-reloadable configuration** — change the role-to-model matrix or provider config without restarting
- **Response caching** — exact-match and semantic near-duplicate caching backed by SQLite

---

## Quickstart

```bash
pip install -e ".[all]"
```

### Python library

```python
from dragonlight_router import RouterEngine

router = RouterEngine()  # loads config/router.yaml

# Ranked models for a role — your app makes the LLM call
models = router.select_models("code_review", top_n=5)
# → ["groq/llama-3.3-70b-versatile", "cerebras/llama3.1-70b", ...]

# Record the outcome so budget + health stay accurate
from dragonlight_router.core.types import RequestOutcome
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
# → { "models": ["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile", ...] }
```

---

## Documentation

| | |
|---|---|
| **Full documentation** | [`docs/`](docs/) |
| Configuration guide | [`docs/live-specs/`](docs/live-specs/) |
| API reference (OpenAPI) | [`docs/openapi.yaml`](docs/openapi.yaml) |
| Architecture | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Deployment runbook | [`docs/deployment-runbook.md`](docs/deployment-runbook.md) |
| Security / hazard register | [`docs/hazard-register.md`](docs/hazard-register.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

---

## Supported providers

Anthropic, Cerebras, Cohere, Gemini, Groq, Mistral, NVIDIA NIM, Ollama, OpenAI, OpenRouter, Together

All providers except Anthropic support automatic catalog refresh. Ollama runs locally and needs no API key. Copy `.env.example` to `.env` and fill in the keys for the providers you use.

---

## Running tests

```bash
pip install -e ".[all,dev]"
python3 -m pytest --no-cov -q          # full suite
```

---

## License

MIT. See [LICENSE](LICENSE).
