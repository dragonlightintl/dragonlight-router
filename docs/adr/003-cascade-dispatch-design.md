# ADR-003: Cascade Dispatch Design (MBR/CBR/LBR)

## Status
Accepted

## Context
Model selection must consider three independent constraint axes: capability (does the model support the required features?), cost (is there budget headroom and is the model healthy?), and rate limits (does the provider have remaining capacity?). A single-stage scoring function that combines all three into one weighted score has two problems: it allows a model with zero rate-limit capacity to score highly on cost/health and get selected (only to fail on dispatch), and it makes it difficult to tune one axis without affecting the others. The constraints also have a natural precedence: there is no point scoring a model that lacks a required capability, and no point checking rate limits on a model that has already been eliminated by budget.

## Decision
Selection is decomposed into three sequential stages:

1. **MBR (Model-Based Ranking)** — filters candidates by capability match (tool use, long context), complexity tier, and operational status (not retired, not circuit-broken). Pure elimination.
2. **CBR (Cost-Based Ranking)** — scores surviving candidates on budget headroom and health using configurable weights. A cost governor dynamically shifts weights under budget pressure. Produces a scored and ranked list.
3. **LBR (Limit-Based Ranking)** — enforces hard rate-limit capacity gates (RPM, RPD, TPM), interleaves candidates across providers to prevent concentration, and applies weighted random selection for the final pick.

After selection, the dispatch layer calls the chosen adapter and walks the fallback chain on failure.

## Consequences
**Positive:**
- Each stage has a single responsibility and can be tested in isolation.
- Capability filtering (MBR) prevents wasted scoring work on ineligible models.
- Rate-limit enforcement (LBR) is a hard gate, not a soft score — prevents dispatch to exhausted providers.
- Provider interleaving in LBR prevents thundering-herd concentration.

**Negative:**
- Three stages add pipeline complexity compared to a single scoring function.
- Inter-stage contracts (candidate list shape) must be kept in sync.
- Tuning requires understanding which stage is responsible for a given behavior.
