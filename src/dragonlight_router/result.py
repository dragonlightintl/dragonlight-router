"""Result type for explicit error handling.

Provides Ok/Err union type similar to Rust's Result for fallible operations.
All functions that can fail should return Result[T, E] where T is the success
type and E is the error type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful result containing a value."""
    
    value: T
    
    def is_ok(self) -> bool:
        return True
    
    def is_err(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        return self.value
    
    def unwrap_err(self) -> E:
        raise AssertionError("Called unwrap_err on Ok value")


@dataclass(frozen=True)
class Err(Generic[E]):
    """Failed result containing an error."""
    
    error: E
    
    def is_ok(self) -> bool:
        return False
    
    def is_err(self) -> bool:
        return True
    
    def unwrap(self) -> T:
        raise AssertionError("Called unwrap on Err value")
    
    def unwrap_err(self) -> E:
        return self.error


Result = Union[Ok[T], Err[E]]


def ok(value: T) -> Ok[T]:
    """Create an Ok result."""
    return Ok(value)


def err(error: E) -> Err[E]:
    """Create an Err result."""
    return Err(error)


def is_ok(result: Result[T, E]) -> bool:
    """Check if result is Ok."""
    return result.is_ok()


def is_err(result: Result[T, E]) -> bool:
    """Check if result is Err."""
    return result.is_err()


def unwrap(result: Result[T, E]) -> T:
    """Extract value from Ok result, panic on Err."""
    return result.unwrap()


def unwrap_err(result: Result[T, E]) -> E:
    """Extract error from Err result, panic on Ok."""
    return result.unwrap_err()