"""Model Spectrography -- direct-adapter profiling for flavor fingerprints.

Public API:
    - probes: SpectrographyProbe definitions and retrieval
    - analyzer: Score aggregation, rank normalization, calibration
    - lifecycle: Staleness, decay, incremental merge, I/O
    - runner: Orchestrator for full spectrography runs
    - storage: SQLite-backed profile persistence (WAL mode)
"""

from dragonlight_router.spectrography.analyzer import (
    ProbeResult,
    aggregate_scores,
    build_fingerprints_yaml,
    build_model_rankings,
    compute_calibration_deltas,
    rank_normalize,
)
from dragonlight_router.spectrography.lifecycle import (
    apply_spectrography_decay,
    check_staleness,
    get_models_needing_spectrography,
    load_existing_fingerprints,
    merge_incremental,
    write_fingerprints_yaml,
)
from dragonlight_router.spectrography.probes import (
    DISCRIMINATION_AXES,
    SpectrographyProbe,
    get_all_probes,
    get_probes_by_axis,
    get_probes_by_domain,
    get_probes_by_task_type,
)
from dragonlight_router.spectrography.runner import run_spectrography
from dragonlight_router.spectrography.storage import SpectrographyStore

__all__ = [
    # Probes
    "DISCRIMINATION_AXES",
    "SpectrographyProbe",
    "get_all_probes",
    "get_probes_by_axis",
    "get_probes_by_domain",
    "get_probes_by_task_type",
    # Analyzer
    "ProbeResult",
    "aggregate_scores",
    "build_fingerprints_yaml",
    "build_model_rankings",
    "compute_calibration_deltas",
    "rank_normalize",
    # Lifecycle
    "apply_spectrography_decay",
    "check_staleness",
    "get_models_needing_spectrography",
    "load_existing_fingerprints",
    "merge_incremental",
    "write_fingerprints_yaml",
    # Runner
    "run_spectrography",
    # Storage
    "SpectrographyStore",
]
