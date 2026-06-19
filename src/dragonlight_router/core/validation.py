"""URL validation utilities for SSRF prevention.

SEC-003 mitigation: Validates provider URLs to prevent Server-Side Request
Forgery attacks. Rejects URLs that resolve to private IP ranges, localhost,
or cloud metadata endpoints.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger()

# Known cloud metadata endpoint hostnames
_METADATA_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "169.254.169.254",
    }
)


def _is_private_ip(host: str) -> bool:
    """Check whether a hostname resolves to a private/reserved IP address.

    Resolves the hostname via DNS and checks all returned addresses against
    private and reserved ranges (10.x, 172.16-31.x, 192.168.x, 127.x,
    169.254.x, ::1, etc.).

    Returns True if any resolved address is private/reserved.
    """
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local
    except ValueError:
        pass

    # Hostname — attempt DNS resolution
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in infos:
            ip_str = sockaddr[0]
            addr = ipaddress.ip_address(ip_str)
            if addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local:
                return True
    except (socket.gaierror, OSError):
        # DNS resolution failed — treat as non-private (will fail at connect time)
        pass

    return False


# DEVIATION CS-004: validate_provider_url is 51 lines (including docstring).
# Justification: The validation is a single linear chain of security checks (hostname,
# metadata, scheme, private IP). Splitting would scatter the SSRF prevention logic and
# make the security review surface harder to audit.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
def validate_provider_url(url: str) -> None:
    """Validate a provider URL for SSRF safety.

    Raises ``ValueError`` if the URL:
    - Uses a scheme other than https (except http for localhost/ollama)
    - Points to a private IP range (10.x, 172.16-31.x, 192.168.x, 127.x, 169.254.x)
    - Contains localhost or cloud metadata endpoints
    - Has no hostname

    The http scheme is allowed for localhost to support local Ollama instances.

    Parameters
    ----------
    url:
        The URL to validate.

    Raises
    ------
    ValueError:
        If the URL fails SSRF validation.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname or ""

    if not hostname:
        raise ValueError(f"URL has no hostname: {url}")

    # Check for metadata endpoints
    if hostname.lower() in _METADATA_HOSTNAMES:
        raise ValueError(f"URL points to cloud metadata endpoint: {url}")

    # Determine if this is a localhost URL (used for local providers like Ollama)
    is_localhost = hostname.lower() in ("localhost", "127.0.0.1", "::1")

    # Scheme validation: https required, except http for localhost
    if scheme == "http":
        if not is_localhost:
            raise ValueError(f"URL scheme must be https for non-localhost URLs: {url}")
        # http://localhost is allowed (e.g., Ollama)
        return
    elif scheme != "https":
        raise ValueError(f"URL scheme must be https (got {scheme}): {url}")

    # For https URLs, check that the host does not resolve to a private IP
    if _is_private_ip(hostname):
        raise ValueError(f"URL resolves to a private/reserved IP address: {url}")
