"""Contract tests for the Result monad.

Verifies that Ok and Err types behave correctly, consistently, and that
the module-level helper functions maintain the same contract.

Spec traceability: Result type contract (core/types.py, result.py)
"""

from __future__ import annotations

import pytest

from dragonlight_router.core.types import Err, Ok, Result
from dragonlight_router.result import err, is_err, is_ok, ok, unwrap, unwrap_err

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Contract: Ok wraps values correctly
# ---------------------------------------------------------------------------


class TestOkContract:
    """Ok must wrap values and expose them correctly."""

    def test_ok_wraps_value(self) -> None:
        """Ok(x).value must be x."""
        result = Ok(42)
        assert result.value == 42

    def test_ok_wraps_string(self) -> None:
        """Ok must work with string values."""
        result = Ok("hello")
        assert result.value == "hello"

    def test_ok_wraps_none(self) -> None:
        """Ok(None) is valid -- None is a legitimate success value."""
        result = Ok(None)
        assert result.value is None

    def test_ok_wraps_complex_type(self) -> None:
        """Ok must work with complex types like dicts."""
        data = {"key": [1, 2, 3]}
        result = Ok(data)
        assert result.value is data

    def test_ok_is_ok_returns_true(self) -> None:
        """Ok.is_ok() must return True."""
        assert Ok(1).is_ok() is True

    def test_ok_is_err_returns_false(self) -> None:
        """Ok.is_err() must return False."""
        assert Ok(1).is_err() is False

    def test_ok_is_frozen(self) -> None:
        """Ok is a frozen dataclass -- mutation must raise."""
        result = Ok(42)
        with pytest.raises((AttributeError, TypeError)):
            result.value = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contract: Err wraps errors correctly
# ---------------------------------------------------------------------------


class TestErrContract:
    """Err must wrap error values and expose them correctly."""

    def test_err_wraps_error(self) -> None:
        """Err(x).error must be x."""
        result = Err("something went wrong")
        assert result.error == "something went wrong"

    def test_err_wraps_exception(self) -> None:
        """Err must work with Exception objects."""
        exc = ValueError("bad value")
        result = Err(exc)
        assert result.error is exc

    def test_err_wraps_complex_type(self) -> None:
        """Err must work with structured error types."""
        error_data = {"code": 404, "message": "not found"}
        result = Err(error_data)
        assert result.error is error_data

    def test_err_is_ok_returns_false(self) -> None:
        """Err.is_ok() must return False."""
        assert Err("x").is_ok() is False

    def test_err_is_err_returns_true(self) -> None:
        """Err.is_err() must return True."""
        assert Err("x").is_err() is True

    def test_err_is_frozen(self) -> None:
        """Err is a frozen dataclass -- mutation must raise."""
        result = Err("error")
        with pytest.raises((AttributeError, TypeError)):
            result.error = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contract: unwrap() behavior
# ---------------------------------------------------------------------------


class TestUnwrapContract:
    """unwrap() on Ok returns value; on Err raises AssertionError."""

    def test_unwrap_ok_returns_value(self) -> None:
        """Ok.unwrap() must return the contained value."""
        assert Ok(42).unwrap() == 42

    def test_unwrap_ok_string(self) -> None:
        """Ok.unwrap() must work with strings."""
        assert Ok("hello").unwrap() == "hello"

    def test_unwrap_err_raises(self) -> None:
        """Err.unwrap() must raise AssertionError."""
        with pytest.raises(AssertionError, match="unwrap on Err"):
            Err("oops").unwrap()

    def test_unwrap_err_on_ok_raises(self) -> None:
        """Ok.unwrap_err() must raise AssertionError."""
        with pytest.raises(AssertionError, match="unwrap_err on Ok"):
            Ok(42).unwrap_err()

    def test_unwrap_err_returns_error(self) -> None:
        """Err.unwrap_err() must return the contained error."""
        assert Err("oops").unwrap_err() == "oops"


# ---------------------------------------------------------------------------
# Contract: Module-level helper functions
# ---------------------------------------------------------------------------


class TestModuleLevelHelpers:
    """result.py helper functions must mirror Ok/Err behavior."""

    def test_ok_helper_creates_ok(self) -> None:
        """ok(x) must return an Ok instance."""
        result = ok(42)
        assert isinstance(result, Ok)
        assert result.value == 42

    def test_err_helper_creates_err(self) -> None:
        """err(x) must return an Err instance."""
        result = err("bad")
        assert isinstance(result, Err)
        assert result.error == "bad"

    def test_is_ok_on_ok(self) -> None:
        """is_ok(Ok(x)) must return True."""
        assert is_ok(Ok(1)) is True

    def test_is_ok_on_err(self) -> None:
        """is_ok(Err(x)) must return False."""
        assert is_ok(Err("x")) is False

    def test_is_err_on_err(self) -> None:
        """is_err(Err(x)) must return True."""
        assert is_err(Err("x")) is True

    def test_is_err_on_ok(self) -> None:
        """is_err(Ok(x)) must return False."""
        assert is_err(Ok(1)) is False

    def test_unwrap_helper_ok(self) -> None:
        """unwrap(Ok(x)) must return x."""
        assert unwrap(Ok(42)) == 42

    def test_unwrap_helper_err_raises(self) -> None:
        """unwrap(Err(x)) must raise AssertionError."""
        with pytest.raises(AssertionError):
            unwrap(Err("bad"))

    def test_unwrap_err_helper_err(self) -> None:
        """unwrap_err(Err(x)) must return x."""
        assert unwrap_err(Err("oops")) == "oops"

    def test_unwrap_err_helper_ok_raises(self) -> None:
        """unwrap_err(Ok(x)) must raise AssertionError."""
        with pytest.raises(AssertionError):
            unwrap_err(Ok(42))


# ---------------------------------------------------------------------------
# Contract: Type union correctness
# ---------------------------------------------------------------------------


class TestResultTypeUnion:
    """Result[T, E] = Ok[T] | Err[E] must hold."""

    def test_ok_is_result(self) -> None:
        """Ok instances must match Result type union."""
        result: Result[int, str] = Ok(42)
        assert isinstance(result, Ok)

    def test_err_is_result(self) -> None:
        """Err instances must match Result type union."""
        result: Result[int, str] = Err("fail")
        assert isinstance(result, Err)

    def test_ok_and_err_are_distinct(self) -> None:
        """Ok and Err must be distinguishable."""
        ok_val = Ok(1)
        err_val = Err(1)
        assert type(ok_val) is not type(err_val)
        assert ok_val.is_ok() != err_val.is_ok()
        assert ok_val.is_err() != err_val.is_err()

    def test_ok_equality(self) -> None:
        """Ok(x) == Ok(x) when values are equal."""
        assert Ok(42) == Ok(42)
        assert Ok("a") == Ok("a")

    def test_err_equality(self) -> None:
        """Err(x) == Err(x) when errors are equal."""
        assert Err("bad") == Err("bad")
        assert Err(42) == Err(42)

    def test_ok_not_equal_err(self) -> None:
        """Ok(x) != Err(x) even when wrapped values are the same."""
        assert Ok(42) != Err(42)
