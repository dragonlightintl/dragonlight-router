# CLI Tools

Dragonlight Router ships with two CLI tools for offline inspection and management. Both read local state files directly and do not require the HTTP server to be running.

## dragonlight-status

Inspect router health, budget, and provider status from the command line.

### Commands

#### `dragonlight-status status`

Show an overall health summary including circuit breaker states, retired models, and budget activity.

```bash
dragonlight-status status
dragonlight-status status --state-dir ./router_state
```

Output includes:

- **Overall availability** -- healthy, degraded, or unavailable based on circuit breaker and retirement state.
- **Circuit breakers** -- per-model state (CLOSED, OPEN, HALF_OPEN) with error counts.
- **Retired models** -- models evicted from routing due to 403/404 errors at inference time.
- **Budget summary** -- daily request and token counts per provider.

#### `dragonlight-status budget`

Show per-provider rate limit usage with RPM, RPD, TPM, and daily token cap breakdowns.

```bash
dragonlight-status budget
dragonlight-status budget --state-dir ./router_state
```

Reads `budget.db` (SQLite) for live sliding-window counts and the provider rate limits from `router.yaml` to compute remaining capacity.

#### `dragonlight-status retired`

List all retired models with their retirement timestamp and age.

```bash
dragonlight-status retired
dragonlight-status retired --state-dir ./router_state
```

Models are retired automatically when the router receives HTTP 403 or 404 at inference time. They can be reinstated via the admin API (`POST /admin/reinstate`).

### Options

All subcommands accept:

| Option | Description |
|---|---|
| `--state-dir PATH` | Path to the router state directory. Defaults to the `state_dir` value from `router.yaml` (typically `./router_state`). |

### State files read

| File | Contents |
|---|---|
| `health_state.json` | Circuit breaker states, error counts, retired models |
| `budget.db` | SQLite request log (RPM/RPD/TPM sliding windows) |
| `budget_state.json` | Daily counters and day-reset boundary |

---

## dragonlight-matrix

Manage the model role matrix -- the ranked mapping of models to routing roles (coding, general, classification, etc.).

### Commands

#### `dragonlight-matrix seed`

Auto-populate the role matrix from the provider catalog.

```bash
dragonlight-matrix seed
dragonlight-matrix seed --state-dir ./router_state --merge
dragonlight-matrix seed --no-merge
```

| Option | Description |
|---|---|
| `--state-dir PATH` | Path to router state directory. |
| `--config PATH` | Path to router config YAML. |
| `--merge / --no-merge` | Preserve existing operator-curated ranks (default: `--merge`). |

#### `dragonlight-matrix update`

Update the role matrix from spectrography profiling results.

```bash
dragonlight-matrix update
dragonlight-matrix update --spectrography-dir ./spectrography_results --blend 0.7
```

| Option | Description |
|---|---|
| `--state-dir PATH` | Path to router state directory. |
| `--spectrography-dir PATH` | Path to spectrography output directory. |
| `--blend FLOAT` | Blend weight for empirical scores (default: 0.7). |

#### `dragonlight-matrix show`

Display the current role matrix in a readable table format.

```bash
dragonlight-matrix show
dragonlight-matrix show --role coding
```

| Option | Description |
|---|---|
| `--state-dir PATH` | Path to router state directory. |
| `--role ROLE` | Filter output to a single role. |

#### `dragonlight-matrix stats`

Show matrix statistics -- unique model count, role breakdown, provider distribution, rank distribution.

```bash
dragonlight-matrix stats
```

| Option | Description |
|---|---|
| `--state-dir PATH` | Path to router state directory. |

#### `dragonlight-matrix profile-pending`

List models that need spectrography profiling (heuristic ranks only).

```bash
dragonlight-matrix profile-pending
dragonlight-matrix profile-pending --limit 10
```

| Option | Description |
|---|---|
| `--state-dir PATH` | Path to router state directory. |
| `--limit N` | Show only the top N models. |
