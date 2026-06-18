"""IBR automated benchmarking system.

Provides the benchmark pipeline for generating model flavor profiles
via standardized eval prompts scored by LLM-as-judge.

Spec reference: intent-based-router-v0.1.0-spec.md section 3.2, Method 3.
"""
from dragonlight_router.benchmark.dogfood import main as dogfood_main
from dragonlight_router.benchmark.dogfood import run_dogfood
from dragonlight_router.benchmark.prompts import EvalPrompt
from dragonlight_router.benchmark.runner import (
    BenchmarkRunner,
    apply_decay,
    run_benchmark_cli,
)

__all__ = [
    "BenchmarkRunner",
    "EvalPrompt",
    "apply_decay",
    "dogfood_main",
    "run_benchmark_cli",
    "run_dogfood",
]
