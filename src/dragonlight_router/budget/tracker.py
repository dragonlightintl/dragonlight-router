1|"""Budget tracker — per-provider rate limit tracking with sliding windows.
     2|
     3|Tracks RPM (requests per minute) via a sliding window of timestamps,
     4|and RPD (requests per day) via a simple counter with daily reset.
     5|"""
     6|from __future__ import annotations
     7|
     8|import datetime as dt
     9|import time
    10|from collections import defaultdict, deque
    11|
    12|import structlog
    13|
    14|from dragonlight_router.core.errors import ProviderNotFoundError
    15|from dragonlight_router.core.types import Err, Ok, ProviderConfig, Result
    16|
    17|logger = structlog.get_logger()
    18|
    19|
    20|def invariant(condition: bool, message: str) -> None:
    21|    """Inline invariant check — enforced even under python -O."""
    22|    if not condition:
    23|        raise AssertionError(message)
    24|
    25|
    26|class BudgetTracker:
    27|    """Tracks rate limit budget for all configured providers."""
    28|
    29|    def __init__(self, providers: list[ProviderConfig]) -> None:
    30|        self._providers: dict[str, ProviderConfig] = {p.name: p for p in providers}
    31|        self._rpm_windows: dict[str, deque[float]] = defaultdict(deque)
    32|        self._rpd_counts: dict[str, int] = defaultdict(int)
    33|        self._day_reset_at: float = self._next_day_boundary()
    34|
    35|    def score(self, provider_name: str) -> Result[float, ProviderNotFoundError]:
    36|        """Budget availability score (0-100) for a provider."""
    37|        provider = self._providers.get(provider_name)
    38|        if provider is None:
    39|            logger.debug("provider_not_found", provider=provider_name)
    40|            return Err(ProviderNotFoundError(provider=provider_name))
    41|
    42|        rpm_remaining = self._rpm_remaining(provider_name)
    43|        rpm_limit = provider.rpm_limit
    44|
    45|        rpd_remaining: int | None = None
    46|        rpd_limit: int | None = provider.rpd_limit
    47|
    48|        if rpd_limit is not None:
    49|            self._maybe_reset_daily()
    50|            rpd_remaining = max(0, rpd_limit - self._rpd_counts[provider_name])
    51|
    52|        rpm_ratio = rpm_remaining / rpm_limit if rpm_limit > 0 else 1.0
    53|
    54|        if rpd_remaining is None or rpd_limit is None or rpd_limit == 0:
    55|            rpd_ratio = 1.0
    56|        else:
    57|            rpd_ratio = rpd_remaining / rpd_limit
    58|
    59|        # Assertions for coding standard (>=2 assertions)
    60|        assert 0.0 <= rpm_ratio <= 1.0, f"rpm_ratio out of bounds: {rpm_ratio}"
    61|        assert 0.0 <= rpd_ratio <= 1.0, f"rpd_ratio out of bounds: {rpd_ratio}"
    62|
    63|        return Ok(min(rpm_ratio, rpd_ratio) * 100.0)
    64|
    65|    def record_request(self, provider_name: str, tokens_used: int = 0) -> None:
    66|        """Record that a request was dispatched."""
    67|        now = time.time()
    68|        self._rpm_windows[provider_name].append(now)
    69|        self._rpd_counts[provider_name] += 1
    70|
    71|    def has_capacity(self, provider_name: str) -> bool:
    72|        """Quick check: does this provider have RPM and RPD headroom?"""
    73|        provider = self._providers.get(provider_name)
    74|        if provider is None:
    75|            return True
    76|        if not self._rpm_remaining(provider_name):
    77|            return False
    78|        return self._rpd_remaining(provider_name) > 0
    79|
    80|    def _rpm_remaining(self, provider_name: str) -> int:
    81|        """Remaining RPM in the current minute window."""
    82|        provider = self._providers.get(provider_name)
    83|        invariant(
    84|            provider is not None,
    85|            f"_rpm_remaining called for unknown provider: {provider_name}",
    86|        )
    87|        limit = provider.rpm_limit
    88|        if limit <= 0:
    89|            return 1
    90|        now = time.time()
    91|        cutoff = now - 60.0
    92|        window = self._rpm_windows.get(provider_name, deque())
    93|        while window and window[0] < cutoff:
    94|            window.popleft()
    95|        return max(0, limit - len(window))
    96|
    97|    def _rpd_remaining(self, provider_name: str) -> int:
    98|        """Remaining RPD in the current day."""
    99|        provider = self._providers.get(provider_name)
   100|        if provider is None or provider.rpd_limit is None:
   101|            return 0
   102|        self._maybe_reset_daily()
   103|        return max(0, provider.rpd_limit - self._rpd_counts[provider_name])
   104|
   105|    def _maybe_reset_daily(self) -> None:
   106|        """Reset daily counters if a day boundary has passed."""
   107|        now = time.time()
   108|        if now >= self._day_reset_at:
   109|            self._rpd_counts.clear()
   110|            self._day_reset_at = self._next_day_boundary()
   111|
   112|    @staticmethod
   113|    def _next_day_boundary() -> float:
   114|        """Compute the next UTC midnight timestamp."""
   115|        tomorrow = dt.datetime.now(dt.UTC).replace(
   116|            hour=0, minute=0, second=0, microsecond=0,
   117|        ) + dt.timedelta(days=1)
   118|        return tomorrow.timestamp()
   119|
   120|    def _tpm_remaining(self, provider_name: str) -> int:
   121|        """Remaining TPM in the current minute window."""
   122|        # Placeholder: TPM tracking not yet implemented (see TM-012).
   123|        return 0
   124|