"""Tests for dragonlight_router.server.openai_proxy — OpenAI-compatible proxy layer.

Covers the translation layer between OpenAI chat completions format and the
router's internal DispatchOrder / EngineResponse types.
"""

from __future__ import annotations

import pytest

from dragonlight_router.core.types import (
    BackendTier,
    DispatchOrder,
    EngineResponse,
)
from dragonlight_router.server.openai_proxy import (
    engine_response_to_openai,
    format_openai_error,
    openai_request_to_dispatch_order,
    parse_model_routing,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_response(**overrides) -> EngineResponse:
    """Build an EngineResponse with sensible defaults, applying overrides."""
    defaults = dict(
        content="Hello, world!",
        backend_used="anthropic/claude-sonnet-4-20250514",
        backend_tier=BackendTier.COMPLEX,
        tokens_in=100,
        tokens_out=50,
        estimated_cost_usd=0.001,
        latency_ms=350.0,
        was_fallback=False,
        fallback_chain=["anthropic/claude-sonnet-4-20250514"],
        dispatch_mode="cascade",
    )
    defaults.update(overrides)
    return EngineResponse(**defaults)


def _minimal_openai_body(
    model: str = "auto",
    messages: list[dict] | None = None,
    **extra,
) -> dict:
    """Build a minimal OpenAI-format request body."""
    body: dict = {
        "model": model,
        "messages": messages
        or [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    body.update(extra)
    return body


# ===========================================================================
# parse_model_routing
# ===========================================================================


class TestParseModelRouting:
    """Tests for parse_model_routing(model_name, headers)."""

    def test_auto_returns_general_defaults(self):
        """'auto' model routes to general intent with mid/standard defaults."""
        pinned, intent, stakes, complexity = parse_model_routing("auto", {})
        assert pinned is None
        assert intent == "general"
        assert stakes == "mid"
        assert complexity == "standard"

    @pytest.mark.parametrize(
        ("model_name", "expected_intent"),
        [
            ("auto-test", "test_generation"),
            ("auto-code", "coding"),
            ("auto-impl", "coding"),
            ("auto-review", "code_review"),
            ("auto-reason", "complex_reasoning"),
            ("auto-debug", "debugging"),
        ],
    )
    def test_auto_variants_map_to_correct_intent(self, model_name, expected_intent):
        """Each auto-* variant maps to the correct intent category."""
        pinned, intent, stakes, complexity = parse_model_routing(model_name, {})
        assert pinned is None
        assert intent == expected_intent
        assert stakes == "mid"
        assert complexity == "standard"

    def test_unknown_auto_variant_falls_back_to_general(self):
        """An unrecognized auto-* variant defaults to 'general'."""
        pinned, intent, stakes, complexity = parse_model_routing("auto-unknown", {})
        assert pinned is None
        assert intent == "general"
        assert stakes == "mid"
        assert complexity == "standard"

    def test_auto_nonsense_suffix_falls_back_to_general(self):
        """A completely made-up auto- suffix defaults to general."""
        pinned, intent, _, _ = parse_model_routing("auto-xyzzy", {})
        assert pinned is None
        assert intent == "general"

    def test_pinned_model_passthrough(self):
        """A non-auto model name is returned as pinned_model."""
        model = "nvidia_nim/deepseek-v4-pro"
        pinned, intent, stakes, complexity = parse_model_routing(model, {})
        assert pinned == model
        assert intent == "general"
        assert stakes == "mid"
        assert complexity == "standard"

    def test_pinned_model_with_slash(self):
        """Provider/model format is preserved exactly."""
        model = "openrouter/meta-llama/llama-3.3-70b"
        pinned, intent, _, _ = parse_model_routing(model, {})
        assert pinned == model

    def test_header_overrides_intent(self):
        """X-Dragonlight-Intent header overrides model-name-derived intent."""
        _, intent, _, _ = parse_model_routing(
            "auto-code",
            {"x-dragonlight-intent": "summarization"},
        )
        assert intent == "summarization"

    def test_header_overrides_stakes(self):
        """X-Dragonlight-Stakes header overrides default 'mid'."""
        _, _, stakes, _ = parse_model_routing(
            "auto",
            {"x-dragonlight-stakes": "high"},
        )
        assert stakes == "high"

    def test_header_overrides_complexity(self):
        """X-Dragonlight-Complexity header overrides default 'standard'."""
        _, _, _, complexity = parse_model_routing(
            "auto",
            {"x-dragonlight-complexity": "complex"},
        )
        assert complexity == "complex"

    def test_all_headers_override_simultaneously(self):
        """All three header overrides can be applied at once."""
        headers = {
            "x-dragonlight-intent": "creative",
            "x-dragonlight-stakes": "low",
            "x-dragonlight-complexity": "trivial",
        }
        pinned, intent, stakes, complexity = parse_model_routing("auto", headers)
        assert pinned is None
        assert intent == "creative"
        assert stakes == "low"
        assert complexity == "trivial"

    def test_header_overrides_ignored_for_pinned_model(self):
        """Pinned models bypass intent routing — headers are not applied."""
        headers = {"x-dragonlight-intent": "coding"}
        pinned, intent, _, _ = parse_model_routing(
            "nvidia_nim/deepseek-v4-pro", headers
        )
        assert pinned == "nvidia_nim/deepseek-v4-pro"
        assert intent == "general"

    def test_empty_headers(self):
        """Empty headers dict uses all defaults."""
        _, intent, stakes, complexity = parse_model_routing("auto", {})
        assert intent == "general"
        assert stakes == "mid"
        assert complexity == "standard"

    def test_lowercase_header_keys_required(self):
        """Starlette normalizes headers to lowercase before passing to handlers."""
        _, intent, _, _ = parse_model_routing(
            "auto",
            {"x-dragonlight-intent": "debugging"},
        )
        assert intent == "debugging"


# ===========================================================================
# openai_request_to_dispatch_order
# ===========================================================================


class TestOpenaiRequestToDispatchOrder:
    """Tests for openai_request_to_dispatch_order(body, headers)."""

    def test_auto_model_produces_unpinned_order(self):
        """model='auto' results in a DispatchOrder with model=None."""
        body = _minimal_openai_body(model="auto")
        order = openai_request_to_dispatch_order(body, {})
        assert isinstance(order, DispatchOrder)
        assert order.model is None

    def test_pinned_model_passes_through(self):
        """A non-auto model name lands in order.model."""
        body = _minimal_openai_body(model="nvidia_nim/foo")
        order = openai_request_to_dispatch_order(body, {})
        assert order.model == "nvidia_nim/foo"

    def test_messages_passed_as_tuple(self):
        """Messages from the request body are stored as a tuple."""
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        body = _minimal_openai_body(messages=msgs)
        order = openai_request_to_dispatch_order(body, {})
        assert isinstance(order.messages, tuple)
        assert len(order.messages) == 2

    def test_system_prompt_extracted(self):
        """system_prompt is extracted from the first system message."""
        body = _minimal_openai_body(
            messages=[
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Ahoy"},
            ]
        )
        order = openai_request_to_dispatch_order(body, {})
        assert order.system_prompt == "You are a pirate."

    def test_operator_message_is_last_user_message(self):
        """operator_message is taken from the last user message."""
        body = _minimal_openai_body(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow up"},
            ]
        )
        order = openai_request_to_dispatch_order(body, {})
        assert order.operator_message == "follow up"

    def test_context_tokens_estimated_from_chars(self):
        """context_tokens is estimated as total chars / 4, minimum 1."""
        # "Hello" = 5 chars in system + "Hi" = 2 chars in user = 7 chars total -> 7 / 4 = 1
        body = _minimal_openai_body(
            messages=[
                {"role": "system", "content": "Hello"},
                {"role": "user", "content": "Hi"},
            ]
        )
        order = openai_request_to_dispatch_order(body, {})
        # 7 chars / 4 = 1 (integer division) — at least 1
        assert order.context_tokens >= 1

    def test_context_tokens_minimum_is_one(self):
        """Even a near-empty message list produces at least 1 context token."""
        body = _minimal_openai_body(
            messages=[{"role": "user", "content": ""}]
        )
        order = openai_request_to_dispatch_order(body, {})
        assert order.context_tokens >= 1

    def test_context_tokens_scales_with_content(self):
        """Longer messages produce proportionally more context tokens."""
        short_body = _minimal_openai_body(
            messages=[{"role": "user", "content": "Hi"}]
        )
        long_content = "x" * 4000
        long_body = _minimal_openai_body(
            messages=[{"role": "user", "content": long_content}]
        )
        short_order = openai_request_to_dispatch_order(short_body, {})
        long_order = openai_request_to_dispatch_order(long_body, {})
        assert long_order.context_tokens > short_order.context_tokens

    def test_tools_passed_through_as_tuple(self):
        """Tools from the request body are stored as a tuple in the order."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        body = _minimal_openai_body(tools=tools)
        order = openai_request_to_dispatch_order(body, {})
        assert isinstance(order.tools, tuple)
        assert len(order.tools) == 1
        assert order.tools[0]["function"]["name"] == "get_weather"

    def test_requires_tool_use_true_when_tools_present(self):
        """requires_tool_use is True when the tools list is non-empty."""
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        body = _minimal_openai_body(tools=tools)
        order = openai_request_to_dispatch_order(body, {})
        assert order.requires_tool_use is True

    def test_requires_tool_use_false_when_no_tools(self):
        """requires_tool_use is False when tools are absent."""
        body = _minimal_openai_body()
        order = openai_request_to_dispatch_order(body, {})
        assert order.requires_tool_use is False

    def test_tools_none_when_absent(self):
        """tools is None when the request body has no tools field."""
        body = _minimal_openai_body()
        order = openai_request_to_dispatch_order(body, {})
        assert order.tools is None

    def test_fallback_policy_from_header(self):
        """x-dragonlight-fallback-policy header sets fallback_policy."""
        body = _minimal_openai_body()
        headers = {"x-dragonlight-fallback-policy": "deny"}
        order = openai_request_to_dispatch_order(body, headers)
        assert order.fallback_policy == "deny"

    def test_fallback_policy_defaults_to_allow(self):
        """Without the header, fallback_policy defaults to 'allow'."""
        body = _minimal_openai_body()
        order = openai_request_to_dispatch_order(body, {})
        assert order.fallback_policy == "allow"

    def test_intent_category_from_model_name(self):
        """auto-code model sets intent_category to 'coding'."""
        body = _minimal_openai_body(model="auto-code")
        order = openai_request_to_dispatch_order(body, {})
        assert order.intent_category == "coding"

    def test_no_system_message_produces_empty_system_prompt(self):
        """When there is no system message, system_prompt is empty."""
        body = _minimal_openai_body(
            messages=[{"role": "user", "content": "Hello"}]
        )
        order = openai_request_to_dispatch_order(body, {})
        assert order.system_prompt == ""

    def test_empty_messages_list_raises(self):
        """An empty messages list raises AssertionError."""
        body = {"model": "auto", "messages": []}
        with pytest.raises(AssertionError):
            openai_request_to_dispatch_order(body, {})

    def test_multimodal_content_array(self):
        """Messages with content as a list (multimodal) are passed through."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            }
        ]
        body = _minimal_openai_body(messages=msgs)
        order = openai_request_to_dispatch_order(body, {})
        assert isinstance(order.messages, tuple)
        # The content structure should be preserved
        assert isinstance(order.messages[0]["content"], list)

    def test_multiple_tools(self):
        """Multiple tools are all passed through."""
        tools = [
            {
                "type": "function",
                "function": {"name": "tool_a", "description": "A", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "tool_b", "description": "B", "parameters": {}},
            },
        ]
        body = _minimal_openai_body(tools=tools)
        order = openai_request_to_dispatch_order(body, {})
        assert len(order.tools) == 2

    def test_tool_use_conversation_messages(self):
        """A multi-turn tool-use conversation is passed through correctly."""
        msgs = [
            {"role": "system", "content": "You can use tools."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Portland"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temp": 72}',
            },
        ]
        body = _minimal_openai_body(messages=msgs)
        order = openai_request_to_dispatch_order(body, {})
        assert len(order.messages) == 4
        assert order.messages[2]["role"] == "assistant"
        assert order.messages[3]["role"] == "tool"


# ===========================================================================
# engine_response_to_openai
# ===========================================================================


class TestEngineResponseToOpenai:
    """Tests for engine_response_to_openai(engine_resp, request_model)."""

    def test_response_id_format(self):
        """Response id starts with 'chatcmpl-dragonlight-'."""
        resp = _make_engine_response()
        result = engine_response_to_openai(resp, "auto")
        assert result["id"].startswith("chatcmpl-dragonlight-")

    def test_object_type(self):
        """object field is 'chat.completion'."""
        resp = _make_engine_response()
        result = engine_response_to_openai(resp, "auto")
        assert result["object"] == "chat.completion"

    def test_created_is_integer(self):
        """created field is an integer timestamp."""
        resp = _make_engine_response()
        result = engine_response_to_openai(resp, "auto")
        assert isinstance(result["created"], int)

    def test_model_is_backend_used(self):
        """model field reflects the actual backend used."""
        resp = _make_engine_response(backend_used="groq/llama-3.3-70b")
        result = engine_response_to_openai(resp, "auto")
        assert result["model"] == "groq/llama-3.3-70b"

    def test_text_response_content(self):
        """Text response: choices[0].message.content matches engine content."""
        resp = _make_engine_response(content="The answer is 42.")
        result = engine_response_to_openai(resp, "auto")
        choice = result["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "The answer is 42."

    def test_text_response_finish_reason_stop(self):
        """Text response has finish_reason='stop'."""
        resp = _make_engine_response(content="Done.", finish_reason="stop")
        result = engine_response_to_openai(resp, "auto")
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_tool_calls_response(self):
        """Tool-call response: tool_calls are placed in the message."""
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            }
        ]
        resp = _make_engine_response(
            content="",
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )
        result = engine_response_to_openai(resp, "auto")
        choice = result["choices"][0]
        assert choice["message"]["tool_calls"] == tool_calls
        assert choice["finish_reason"] == "tool_calls"

    def test_usage_tokens(self):
        """usage block has correct prompt_tokens and completion_tokens."""
        resp = _make_engine_response(tokens_in=200, tokens_out=80)
        result = engine_response_to_openai(resp, "auto")
        assert result["usage"]["prompt_tokens"] == 200
        assert result["usage"]["completion_tokens"] == 80

    def test_usage_total_tokens(self):
        """usage.total_tokens is the sum of prompt + completion tokens."""
        resp = _make_engine_response(tokens_in=200, tokens_out=80)
        result = engine_response_to_openai(resp, "auto")
        assert result["usage"]["total_tokens"] == 280

    def test_dragonlight_metadata_present(self):
        """dragonlight metadata block is present with routing details."""
        resp = _make_engine_response(
            backend_used="anthropic/claude-sonnet-4-20250514",
            backend_tier=BackendTier.COMPLEX,
            dispatch_mode="cascade",
            was_fallback=False,
        )
        result = engine_response_to_openai(resp, "auto")
        dl = result["dragonlight"]
        assert dl["backend_used"] == "anthropic/claude-sonnet-4-20250514"
        assert dl["dispatch_mode"] == "cascade"
        assert dl["was_fallback"] is False

    def test_dragonlight_backend_tier(self):
        """dragonlight.backend_tier reflects the tier of the backend used."""
        resp = _make_engine_response(backend_tier=BackendTier.SIMPLE)
        result = engine_response_to_openai(resp, "auto")
        # The tier could be serialized as the enum value string or the enum itself
        tier_value = result["dragonlight"]["backend_tier"]
        # Accept either "simple" string or BackendTier.SIMPLE enum
        assert tier_value in ("simple", BackendTier.SIMPLE)

    def test_system_fingerprint_present(self):
        """system_fingerprint is present in the response."""
        resp = _make_engine_response()
        result = engine_response_to_openai(resp, "auto")
        assert "system_fingerprint" in result

    def test_fallback_response_metadata(self):
        """was_fallback=True is reflected in the dragonlight metadata."""
        resp = _make_engine_response(
            was_fallback=True,
            fallback_chain=["anthropic/claude-sonnet-4-20250514", "groq/llama-3.3-70b"],
        )
        result = engine_response_to_openai(resp, "auto")
        assert result["dragonlight"]["was_fallback"] is True

    def test_choices_has_index_zero(self):
        """choices[0] has index=0."""
        resp = _make_engine_response()
        result = engine_response_to_openai(resp, "auto")
        assert result["choices"][0]["index"] == 0

    def test_single_choice_returned(self):
        """Only one choice is returned (n=1 only)."""
        resp = _make_engine_response()
        result = engine_response_to_openai(resp, "auto")
        assert len(result["choices"]) == 1

    def test_empty_content_text_response(self):
        """An empty-content text response is represented correctly."""
        resp = _make_engine_response(content="", finish_reason="stop")
        result = engine_response_to_openai(resp, "auto")
        assert result["choices"][0]["message"]["content"] == ""
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_tool_calls_with_content(self):
        """A response with both content and tool_calls is handled."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
        resp = _make_engine_response(
            content="Let me check that.",
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )
        result = engine_response_to_openai(resp, "auto")
        choice = result["choices"][0]
        assert choice["message"]["content"] == "Let me check that."
        assert choice["message"]["tool_calls"] == tool_calls
        assert choice["finish_reason"] == "tool_calls"


# ===========================================================================
# format_openai_error
# ===========================================================================


class TestFormatOpenaiError:
    """Tests for format_openai_error(message, error_type, status_code, ...)."""

    def test_error_structure(self):
        """Returns a JSONResponse with the OpenAI error format."""
        resp = format_openai_error(
            message="Model not found",
            error_type="invalid_request_error",
            status_code=404,
        )
        body = resp.body
        # JSONResponse stores its body as bytes; decode to verify structure
        import json

        data = json.loads(body)
        assert "error" in data
        assert data["error"]["message"] == "Model not found"
        assert data["error"]["type"] == "invalid_request_error"

    def test_status_code_set(self):
        """Status code on the JSONResponse matches the provided value."""
        resp = format_openai_error(
            message="Rate limited",
            error_type="rate_limit_error",
            status_code=429,
        )
        assert resp.status_code == 429

    def test_400_bad_request(self):
        """400 status code for invalid request."""
        resp = format_openai_error(
            message="Invalid model",
            error_type="invalid_request_error",
            status_code=400,
        )
        assert resp.status_code == 400

    def test_500_internal_error(self):
        """500 status code for internal errors."""
        resp = format_openai_error(
            message="Internal error",
            error_type="internal_error",
            status_code=500,
        )
        assert resp.status_code == 500
        import json

        data = json.loads(resp.body)
        assert data["error"]["type"] == "internal_error"

    def test_error_contains_param_and_code_fields(self):
        """OpenAI error format includes param and code fields."""
        resp = format_openai_error(
            message="Bad param",
            error_type="invalid_request_error",
            status_code=400,
        )
        import json

        data = json.loads(resp.body)
        err = data["error"]
        assert "param" in err
        assert "code" in err

    def test_error_with_optional_param(self):
        """param field is included in the error when provided."""
        resp = format_openai_error(
            message="Invalid value for model",
            error_type="invalid_request_error",
            status_code=400,
            param="model",
        )
        import json

        data = json.loads(resp.body)
        assert data["error"]["param"] == "model"

    def test_error_with_optional_code(self):
        """code field is included in the error when provided."""
        resp = format_openai_error(
            message="Budget exhausted",
            error_type="billing_error",
            status_code=402,
            code="budget_exhausted",
        )
        import json

        data = json.loads(resp.body)
        assert data["error"]["code"] == "budget_exhausted"

    def test_error_param_defaults_to_none(self):
        """param defaults to None when not provided."""
        resp = format_openai_error(
            message="Error",
            error_type="internal_error",
            status_code=500,
        )
        import json

        data = json.loads(resp.body)
        assert data["error"]["param"] is None

    def test_error_code_defaults_to_none(self):
        """code defaults to None when not provided."""
        resp = format_openai_error(
            message="Error",
            error_type="internal_error",
            status_code=500,
        )
        import json

        data = json.loads(resp.body)
        assert data["error"]["code"] is None
