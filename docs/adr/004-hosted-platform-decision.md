# ADR-004: Hosted Platform — Why Not (Now or Likely Ever)

## Status
Accepted

## Date
2026-06-19

## Context
With dragonlight-router reaching v0.3.0 (100% test coverage, 11 provider adapters, full cascade pipeline, security hardening, CI matrix), the question arose: should Dragonlight International build and launch a hosted version of the router as a commercial SaaS/PaaS product?

The hosted LLM routing market already has multiple funded entrants:
- **OpenRouter** — unified API for 200+ models, auto-routing, pay-per-token with markup
- **Portkey** — AI gateway with routing, fallbacks, caching, guardrails, observability ($3M+ funded)
- **Martian** — intelligent model routing via ML, VC-backed
- **LiteLLM** — open-source proxy (15k+ GitHub stars) plus hosted LiteLLM Enterprise
- **Unify AI** — model routing with quality/cost optimization, funded
- **Not Diamond** — ML-based model routing, funded

## Decision
**Do not build a hosted dragonlight-router platform.** The router's value is as an open-source library, a bespoke consulting deliverable, and a foundational component of the DAOS architecture — not as a multi-tenant hosted service.

### Rationale

**1. The sovereignty narrative contradicts hosted multi-tenancy.**
The router's strongest differentiator is that routing intelligence stays in the operator's stack. Trust tiers, IBR feedback loops, learned flavor profiles, cost governance — these are designed to be operator-owned. A hosted version says "trust us with your routing intelligence instead," which is exactly what OpenRouter and Portkey already offer. Competing on their terms while abandoning our philosophical differentiator is a losing position.

**2. No defensible moat in the hosted market.**
Six funded competitors have years of head start, dedicated engineering teams, and millions of requests per day generating training signal. The cascade architecture, spectrography, and IBR feedback loops are genuine differentiators — but they're differentiators for *bespoke deployments tuned to specific operator needs*, not for a generic hosted platform competing on breadth and uptime SLAs.

**3. Operational burden is incompatible with current resources.**
A hosted routing platform requires:
- $850–3,000/month infrastructure before the first customer pays
- 3–6 months to build multi-tenant auth, billing, onboarding
- 24/7 ops obligation (the proxy sits on every customer's critical path)
- SOC 2 or equivalent compliance for enterprise buyers ($10–50k)
- Continuous adapter maintenance as providers change APIs

Every hour spent on platform ops is an hour stolen from the first consulting sale, which is this season's highest priority.

**4. The router's differentiators are maximally valuable in bespoke deployments.**
Model Spectrography profiles tuned per-client, IBR feedback loops trained on a specific operator's quality ratings, trust tiers mapped to a specific compliance posture — these become genuinely differentiated when customized. A multi-tenant hosted version flattens them into generic features.

### Viable monetization paths (in priority order)

1. **Portfolio piece** — demonstrates technical depth to prospective consulting clients at zero incremental cost.
2. **Bespoke deployment as consulting deliverable** — client pays for a dragonlight-router instance customized to their stack with spectrography profiles tuned to their use cases. Recurring revenue via maintenance/tuning contracts.
3. **Open-source traction → commercial extensions** — premium modules (advanced spectrography, enterprise integrations, compliance features), sold as licenses not hosting.
4. **Managed deployment service** — only if multiple consulting clients independently request it. "We deploy and manage your instance in your cloud," not multi-tenant SaaS.

## Consequences

**Positive:**
- Focus stays on consulting revenue (first sale) and open-source release
- The router serves its designed purpose: sovereignty layer for DAOS and infrastructure for consulting engagements
- No ongoing infrastructure cost or ops burden
- Philosophical coherence between the product's design (operator sovereignty) and its go-to-market

**Negative:**
- Forecloses a potential revenue stream (revisit in 12–18 months if consulting clients request hosted option)
- No recurring SaaS revenue from the router itself

**Revisit conditions:**
- Multiple consulting clients independently request a managed/hosted option
- Consulting revenue is consistent enough to fund 6–12 months of platform development
- Open-source traction creates a natural funnel (500+ GitHub stars, community contributions)
