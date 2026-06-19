"""Result type for explicit error handling.

Provides Ok/Err union type similar to Rust's Result for fallible operations.
All functions that can fail should return Result[T, E] where T is the success
type and E is the error type.
"""

from __future__ import annotations

from typing import TypeVar

from dragonlight_router.core.types import Err, Ok, Result

__all__ = ["Ok", "Err", "Result", "ok", "err", "is_ok", "is_err", "unwrap", "unwrap_err"]

T = TypeVar("T")
E = TypeVar("E")


def ok(value: T) -> Ok[T]:
    """Create an Ok result."""
    result = Ok(value)
    assert isinstance(result, Ok), "ok must return an Ok instance"
    assert result.value == value, "ok must preserve the value"
    return result


def err(error: E) -> Err[E]:
    """Create an Err result."""
    result = Err(error)
    assert isinstance(result, Err), "err must return an Err instance"
    assert result.error == error, "err must preserve the error"
    return result


def is_ok(result: Result[T, E]) -> bool:
    """Check if result is Ok."""
    assert result is not None, "result must not be None"
    ok_result = result.is_ok()
    assert isinstance(ok_result, bool), "is_ok must return a bool"
    return ok_result


def is_err(result: Result[T, E]) -> bool:
    """Check if result is Err."""
    assert result is not None, "result must not be None"
    err_result = result.is_err()
    assert isinstance(err_result, bool), "is_err must return a bool"
    return err_result


def unwrap(result: Result[T, E]) -> T:
    """Extract value from Ok result, panic on Err."""
    assert result is not None, "result must not be None"
    if result.is_ok():
        value = result.unwrap()
        # Cannot assert much about T without knowing it — skip.
        return value
    else:
        raise AssertionError("Called unwrap on Err value")


def unwrap_err(result: Result[T, E]) -> E:
    """Extract error from Err result, panic on Ok."""
    assert result is not None, "result must not be None"
    if result.is_err():
        error = result.unwrap_err()
        return error
    else:
        raise AssertionError("Called unwrap_err on Ok value")
