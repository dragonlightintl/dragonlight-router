"""Dragonlight Router — multi-provider intelligent routing."""

from .core.types import (
    DispatchOrder,
    EngineResponse,
    RequestOutcome,
    StreamChunk,
)
from .result import Err, Ok
from .router import RouterEngine, get_router

__version__ = "0.3.0"

__all__ = [
    "RouterEngine",
    "get_router",
    "__version__",
    # Consumer types
    "DispatchOrder",
    "EngineResponse",
    "RequestOutcome",
    "StreamChunk",
    # Result types
    "Ok",
    "Err",
]
