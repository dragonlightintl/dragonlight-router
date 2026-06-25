# Fianna Audit — Dragonlight Router (2026-06-24)

Auditors: DIAN CECHT (Security), FIRINNE (Ground Truth), GOIBNIU (Engineering Standards), LUGH (Interface & Docs)

## Composite Scores

| Persona | Dimension | Score |
|---------|-----------|-------|
| DIAN CECHT | Security | 0 critical, 2 high, 4 medium, 3 low |
| FIRINNE | Ground Truth | 5/8 VERIFIED, 3/8 PARTIAL |
| GOIBNIU | Engineering Standards | 81/100 |
| LUGH | Interface & Documentation | 82/100 |

---

## Findings Tracker

### WAVE 1 (complete)

| ID | Source | Severity | Finding | Status |
|----|--------|----------|---------|--------|
| PUB-01 | LUGH | BLOCKING | `router_state/` committed to git | DONE (already untracked) |
| PUB-02 | LUGH | BLOCKING | CODE_OF_CONDUCT.md missing | DONE (ead2fd5) |
| PUB-03 | LUGH | BLOCKING | 14+ dev artifacts in repo root | DONE (761fc0a, 17 removed) |
| PUB-04 | LUGH | BLOCKING | SECURITY.md version table stale (0.2.x) | DONE (74bdf0c) |
| SEC-M2 | DIAN CECHT | MEDIUM | HTTP error responses leak internal state | DONE (1e9b2c0) |
| SEC-M1 | DIAN CECHT | MEDIUM | Admin endpoints default-open | DONE (1e9b2c0, fail-closed + admin_open flag) |
| SEC-M3 | DIAN CECHT | MEDIUM | spectrograph_feedback.db world-readable (0644) | DONE (1e9b2c0, chmod 0600 on create) |
| BUG-01 | FIRINNE | PARTIAL | score_candidate() drops spectrograph_match weight | DONE (56cc2ef) |
| DOC-01 | LUGH | RECOMMENDED | Re-export consumer types from __init__.py | DONE (5c5f241) |
| DOC-02 | LUGH | RECOMMENDED | Document IBR + pinned dispatch config | DONE (5c5f241) |
| DOC-03 | LUGH | RECOMMENDED | Add docstrings to get_router() and RouterEngine.__init__ | DONE (5c5f241) |
| DOC-04 | LUGH | RECOMMENDED | Enable mkdocstrings in mkdocs.yml | DONE (5c5f241) |

### WAVE 2 (complete)

| ID | Source | Severity | Finding | Status |
|----|--------|----------|---------|--------|
| ENG-01 | GOIBNIU | HIGH | cascade.py 1,729→1,357 lines — pinned dispatch extracted | DONE (9b1a3c0) |
| ENG-02 | GOIBNIU | HIGH | routes.py 1,564→1,132 lines — OpenAPI schema extracted | DONE (9b1a3c0) |
| ENG-03 | GOIBNIU | MEDIUM | 66 functions exceed 40-line limit without deviation records | DEFERRED (reduced by extraction) |
| ENG-04 | GOIBNIU | MEDIUM | No Hypothesis profiles (local/ci/nightly/debug) in conftest.py | DONE (7c4ab9f) |
| ENG-05 | GOIBNIU | MEDIUM | No domain strategy library (tests/strategies/) | DONE (b60cba0) |
| ENG-06 | GOIBNIU | MEDIUM | No @example pinned cases in property tests | DONE (7c4ab9f, 5 pinned) |
| ENG-07 | GOIBNIU | LOW | No pre-commit hooks (.pre-commit-config.yaml) | DONE (8bbc2cf) |
| ENG-08 | GOIBNIU | LOW | No [tool.ruff] section in pyproject.toml | DONE (8bbc2cf, already existed + updated) |
| SEC-H1 | DIAN CECHT | HIGH | `random` module used in LBR selection — replace with `secrets` | DONE (4197da9) |
| SEC-H2 | DIAN CECHT | HIGH | Supply chain hash pinning not implemented (SEC-SUPPLY-001) | DONE (8bc0d42) |
| SEC-M4 | DIAN CECHT | MEDIUM | model_role_matrix.json loaded without schema validation | DONE (8bbc2cf) |
| SEC-L1 | DIAN CECHT | LOW | No Bandit skip justification for B603 | DONE (7cd6aa9, skip removed — no subprocess usage) |
| SEC-L3 | DIAN CECHT | LOW | CORS defaults allow all headers when enabled | DONE (c09756b) |
| CLI-01 | LUGH | RECOMMENDED | No CLI for health/budget/status inspection without server | DONE (9434b5a) |
| CLI-02 | LUGH | RECOMMENDED | No CLI documentation page in MkDocs | DONE (9434b5a, docs/cli.md) |
| DOC-05 | LUGH | RECOMMENDED | Spectrography user guide missing | DONE (e624dab, docs/spectrography.md) |
| PUB-05 | LUGH | BLOCKING | No PyPI Trusted Publishing release workflow (OSS-050) | DONE (1a6bb37) |
| PUB-06 | LUGH | RECOMMENDED | Version declared in two places (pyproject.toml + __init__.py) | NOTED |

### WAVE 3 (future)

| ID | Source | Severity | Finding | Status |
|----|--------|----------|---------|--------|
| SPEC-01 | GOIBNIU | MEDIUM | Live spec obsolete (v0.1.0) — router is v0.3.0 | PARTIAL (v0.1.0 spec marked as historical; full v0.3.0 rewrite deferred) |
| GT-01 | FIRINNE | PARTIAL | IBR disabled by default — spectrograph scoring dormant | PENDING |
| GT-02 | FIRINNE | INFO | Cascade compensates for scoring gap externally | DONE (BUG-01 fixed in 56cc2ef) |
| ENG-09 | GOIBNIU | LOW | AC traceability sparse in test names | PENDING |
| ENG-10 | GOIBNIU | LOW | No composition tests directory | PENDING |
| ENG-11 | GOIBNIU | LOW | _MODEL_COSTS hardcoded in router.py — should move to config | PENDING |
| SEC-L2 | DIAN CECHT | LOW | _dispatch_cache module-level mutable global | NOTED (deviation documented) |

---

## DIAN CECHT — Full Security Report

### HIGH Findings

**H-1: `random` module used for security-adjacent operations**
- Files: `adapters/_openai_compat.py:149`, `health/circuit_breaker.py:84`, `selection/lbr.py:207`
- `random` PRNG used for retry jitter and weighted selection. `lbr.py:207` uses `random.choices()` for final candidate selection.
- Remediation: Replace with `secrets` module or document as non-security-sensitive with `# nosec`.
- Compliance: Section 1.2 (Cryptographic Failures) — PARTIAL

**H-2: Supply chain hash pinning not implemented**
- File: `pyproject.toml:102-107`
- Dependencies version-pinned but `--require-hashes` not enforced. TODO `SEC-SUPPLY-001` exists.
- Compliance: Security-spec section 6.5 — NON-COMPLIANT

### MEDIUM Findings

**M-1: Admin endpoints unprotected by default**
- `admin_api_key` defaults to None, admin endpoints open. Startup warning logged but access allowed.
- Compliance: Section 1.4 (Insecure Design) — PARTIAL

**M-2: Error messages may leak internal state**
- `str(exc)` included in HTTP response `error_message`/`error_details`. Log scrubber covers logs but not responses.
- Compliance: Security-spec Appendix A — NON-COMPLIANT for HTTP responses

**M-3: spectrograph_feedback.db world-readable (0644)**
- All other state files 0600. Feedback DB is 0644.
- Compliance: Section 4.1/6.4 — PARTIAL

**M-4: model_role_matrix.json unvalidated on load**
- No schema validation. Malicious matrix could manipulate routing.
- Compliance: Section 1.8 (Data Integrity) — PARTIAL

### LOW Findings

**L-1:** B603 Bandit skip without justification. **L-2:** `_dispatch_cache` module-level mutable (documented deviation). **L-3:** CORS defaults allow all headers.

### PASS

- API keys from env vars, never hardcoded (Section 3.1)
- SSRF validation with DNS resolution, private IP blocking, metadata endpoint blocking (Section 1.10)
- Secret scrubbing in logs covers Bearer, sk-, gsk_, xai-, key-, nvapi-, AIza- patterns (Section 6.4)
- TLS via httpx defaults, no verify=False (Network)

---

## FIRINNE — Ground Truth Verdicts

| Capability | Verdict | Notes |
|------------|---------|-------|
| MBR | VERIFIED | Capability-tier filtering with graceful upgrade, no-downgrade invariant |
| CBR | PARTIAL | spectrograph_match weight declared but dropped in _apply_weights() |
| LBR | VERIFIED | Hard capacity gate, median-score threshold, weighted random selection |
| IBR | PARTIAL | Real infrastructure, tested, but disabled by default |
| Cascade | VERIFIED | Full MBR→IBR→CBR→LBR pipeline, fallback chain, pinned dispatch bypass |
| Shared-state budget | VERIFIED | SQLite WAL, cross-process atomic check-and-reserve, RPM/RPD/TPM tracking |
| Spectrography | PARTIAL | Scoring math correct, depends on IBR being enabled (disabled by default) |
| Health tracking | VERIFIED | Circuit breakers, retirement, state persistence, background probes |

---

## GOIBNIU — Engineering Scores

| Dimension | Score |
|-----------|-------|
| Code Quality | 72/100 |
| Testing Standards | 78/100 |
| Architecture | 85/100 |
| Development Pipeline | 88/100 |
| Git Hygiene | 82/100 |
| **Weighted Total** | **81/100** |

Three-stroke plan: (1) Extract pinned dispatch to dispatch/pinned.py, (2) PBT strategy library + Hypothesis profiles, (3) Pre-commit hooks + push.

---

## LUGH — Interface Scores

| Dimension | Score |
|-----------|-------|
| Public API Surface | 82/100 |
| Configuration UX | 88/100 |
| CLI | 75/100 |
| Documentation | 85/100 |
| Open Source Readiness | 72/100 |
| Error Messages | 90/100 |
| **Composite** | **82/100** |
