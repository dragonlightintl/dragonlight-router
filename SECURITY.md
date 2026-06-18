# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report privately via one of:

- [GitHub Security Advisories](https://github.com/dragonlightintl/dragonlight-router/security/advisories/new)
- Email: security@dragonlightintl.com

Include: affected version, reproduction steps, and potential impact.

## Response timeline

| Stage | Target |
|-------|--------|
| Acknowledgement | 72 hours |
| Triage and severity assessment | 7 days |
| Fix or mitigation | 30 days for critical/high |

## Scope

- API key exposure or logging
- Authentication bypass in the HTTP server
- Privilege escalation via configuration loading
- Path traversal in cache or state file writes
- SSRF via provider URL configuration

## Out of scope

- Vulnerabilities in upstream LLM provider APIs
- Issues requiring physical access to the host machine
- Denial of service via intentionally malformed configuration files
