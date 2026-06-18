# ADR-002: Provider Adapter Pattern

## Status
Accepted

## Context
The router integrates with 11 LLM providers (Groq, Cerebras, OpenAI, Anthropic, Gemini, Mistral, NVIDIA NIM, OpenRouter, Cohere, Together, Ollama). Each has its own SDK or HTTP API, but the majority follow the OpenAI chat-completion wire format with minor variations (different auth headers, response envelope shapes, or streaming chunk formats). Without a common interface, the dispatch layer would need provider-specific branching at every call site, and adding a new provider would require changes throughout the codebase.

## Decision
All providers implement the `GenerativeBackend` protocol defined in `core.types`, which specifies `generate()` and `generate_stream()` async methods accepting a `DispatchOrder` and returning `Result[EngineResponse, ...]` or an async iterator of `StreamChunk`. An `_openai_compat.py` base class in the `adapters` package implements the common OpenAI-format request/response handling. Provider-specific adapters inherit from this base and override only what diverges (e.g., Anthropic uses a different message format, Google uses a distinct REST API). Fully compatible providers (Groq, Cerebras, NVIDIA NIM) inherit with near-zero overrides.

## Consequences
**Positive:**
- Adding a new OpenAI-compatible provider requires only a thin subclass (~20 lines).
- The dispatch layer is provider-agnostic — it calls `backend.generate()` uniformly.
- Shared retry, timeout, and error-mapping logic lives in one place.

**Negative:**
- Providers with fundamentally different APIs (Anthropic, Cohere) still need substantial override code.
- The base class carries assumptions (chat-completion format) that may not hold for future non-chat endpoints.
