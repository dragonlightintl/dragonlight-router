"""Tests for dragonlight_router.result — Ok/Err wrapper functions.

Spec traceability: TM-010 (Result type and explicit error handling)
"""

from __future__ import annotations

import pytest

from dragonlight_router.core.types import Err, Ok
from dragonlight_router.result import err, is_err, is_ok, ok, unwrap, unwrap_err

pytestmark = pytest.mark.unit


class TestOk:
    def test_ok_creates_ok_instance(self):
        """[TM-010 AC-1] ok() returns an Ok wrapping the given value."""
        result = ok(42)
        assert isinstance(result, Ok)

    def test_ok_preserves_value(self):
        """[TM-010 AC-1] ok() preserves the exact value passed in."""
        result = ok("hello")
        assert result.value == "hello"

    def test_ok_with_none(self):
        """[TM-010 AC-1] ok() accepts None as a value."""
        result = ok(None)
        assert isinstance(result, Ok)
        assert result.value is None

    def test_ok_with_empty_string(self):
        """[TM-010 AC-1] ok() accepts empty string."""
        result = ok("")
        assert isinstance(result, Ok)
        assert result.value == ""

    def test_ok_with_complex_object(self):
        """[TM-010 AC-1] ok() preserves identity of complex objects."""
        obj = {"key": [1, 2, 3]}
        result = ok(obj)
        assert result.value is obj


class TestErr:
    def test_err_creates_err_instance(self):
        """[TM-010 AC-2] err() returns an Err wrapping the given error."""
        result = err("something went wrong")
        assert isinstance(result, Err)

    def test_err_preserves_error(self):
        """[TM-010 AC-2] err() preserves the exact error passed in."""
        result = err("bad input")
        assert result.error == "bad input"

    def test_err_with_none(self):
        """[TM-010 AC-2] err() accepts None as an error value."""
        result = err(None)
        assert isinstance(result, Err)
        assert result.error is None

    def test_err_with_empty_string(self):
        """[TM-010 AC-2] err() accepts empty string as an error."""
        result = err("")
        assert isinstance(result, Err)
        assert result.error == ""

    def test_err_with_complex_object(self):
        """[TM-010 AC-2] err() preserves identity of complex error objects."""
        obj = {"code": 404, "detail": "not found"}
        result = err(obj)
        assert result.error is obj


class TestIsOk:
    def test_is_ok_true_for_ok(self):
        """[TM-010 AC-3] is_ok() returns True for an Ok result."""
        assert is_ok(ok(1)) is True

    def test_is_ok_false_for_err(self):
        """[TM-010 AC-3] is_ok() returns False for an Err result."""
        assert is_ok(err("oops")) is False

    def test_is_ok_returns_bool(self):
        """[TM-010 AC-3] is_ok() return value is exactly a bool."""
        result = is_ok(ok("x"))
        assert type(result) is bool


class TestIsErr:
    def test_is_err_true_for_err(self):
        """[TM-010 AC-4] is_err() returns True for an Err result."""
        assert is_err(err("bad")) is True

    def test_is_err_false_for_ok(self):
        """[TM-010 AC-4] is_err() returns False for an Ok result."""
        assert is_err(ok(99)) is False

    def test_is_err_returns_bool(self):
        """[TM-010 AC-4] is_err() return value is exactly a bool."""
        result = is_err(err("x"))
        assert type(result) is bool


class TestUnwrap:
    def test_unwrap_extracts_value_from_ok(self):
        """[TM-010 AC-5] unwrap() returns the value from an Ok result."""
        assert unwrap(ok(7)) == 7

    def test_unwrap_extracts_none_from_ok(self):
        """[TM-010 AC-5] unwrap() returns None when Ok contains None."""
        assert unwrap(ok(None)) is None

    def test_unwrap_extracts_complex_object(self):
        """[TM-010 AC-5] unwrap() preserves identity of complex Ok values."""
        obj = [1, 2, 3]
        assert unwrap(ok(obj)) is obj

    def test_unwrap_raises_on_err(self):
        """[TM-010 AC-5] unwrap() raises AssertionError when called on an Err."""
        with pytest.raises(AssertionError):
            unwrap(err("failure"))

    def test_unwrap_raises_assertion_error_not_other(self):
        """[TM-010 AC-5] unwrap() raises specifically AssertionError, not a generic exception."""
        with pytest.raises(AssertionError, match="unwrap"):
            unwrap(err("failure"))


class TestUnwrapErr:
    def test_unwrap_err_extracts_error_from_err(self):
        """[TM-010 AC-6] unwrap_err() returns the error from an Err result."""
        assert unwrap_err(err("bad")) == "bad"

    def test_unwrap_err_extracts_none_error(self):
        """[TM-010 AC-6] unwrap_err() returns None when Err contains None."""
        assert unwrap_err(err(None)) is None

    def test_unwrap_err_extracts_complex_error(self):
        """[TM-010 AC-6] unwrap_err() preserves identity of complex error objects."""
        obj = {"code": 500}
        assert unwrap_err(err(obj)) is obj

    def test_unwrap_err_raises_on_ok(self):
        """[TM-010 AC-6] unwrap_err() raises AssertionError when called on an Ok."""
        with pytest.raises(AssertionError):
            unwrap_err(ok("success"))

    def test_unwrap_err_raises_assertion_error_not_other(self):
        """[TM-010 AC-6] unwrap_err() raises specifically AssertionError, not generic."""
        with pytest.raises(AssertionError, match="unwrap_err"):
            unwrap_err(ok("success"))
