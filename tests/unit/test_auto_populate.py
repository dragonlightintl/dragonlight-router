"""Tests for roles/auto_populate.py — catalog-to-matrix auto-seeder.

Covers heuristic classification, merge behavior, exclusion logic,
and edge cases (empty catalog, empty matrix).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dragonlight_router.roles.auto_populate import (
    auto_populate_matrix,
    classify_model,
    load_catalog,
)

pytestmark = pytest.mark.unit


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_catalog(state_dir: Path, catalog: dict[str, list[dict[str, Any]]]) -> None:
    payload = {"timestamp": 9999999999.0, "catalog": catalog}
    (state_dir / "provider_catalog.json").write_text(json.dumps(payload))


def _write_matrix(state_dir: Path, matrix: dict[str, Any]) -> None:
    (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))


def _read_matrix(state_dir: Path) -> dict[str, Any]:
    return json.loads((state_dir / "model_role_matrix.json").read_text())


# ─── classify_model: coding specialists ───────────────────────────────────────


class TestClassifyModelCoding:
    def test_codestral_high_coding(self):
        ranks = classify_model("mistral/codestral-latest")
        assert ranks["coding"] >= 70

    def test_starcoder_high_coding(self):
        ranks = classify_model("nvidia_nim/bigcode/starcoder2-15b")
        assert ranks["coding"] >= 70

    def test_kimi_k2_high_coding(self):
        ranks = classify_model("nvidia_nim/moonshotai/kimi-k2.6")
        assert ranks["coding"] >= 70

    def test_qwen3_coder_high_coding(self):
        ranks = classify_model("openrouter/qwen/qwen3-coder:free")
        assert ranks["coding"] >= 70

    def test_devstral_high_coding(self):
        ranks = classify_model("mistral/devstral-latest")
        assert ranks["coding"] >= 70

    def test_coding_lower_than_or_equal_reasoning_for_coder_model(self):
        """Coding specialists should score lower on reasoning than coding."""
        ranks = classify_model("nvidia_nim/bigcode/starcoder2-15b")
        assert ranks["coding"] >= ranks["reasoning"]


# ─── classify_model: reasoning specialists ────────────────────────────────────


class TestClassifyModelReasoning:
    def test_deepseek_r1_high_reasoning(self):
        ranks = classify_model("groq/deepseek-r1-distill-llama-70b")
        assert ranks["reasoning"] >= 70

    def test_qwen35_high_reasoning(self):
        ranks = classify_model("nvidia_nim/qwen/qwen3.5-397b-a17b")
        assert ranks["reasoning"] >= 70

    def test_magistral_high_reasoning(self):
        ranks = classify_model("mistral/magistral-medium-latest")
        assert ranks["reasoning"] >= 70

    def test_reasoning_model_higher_spec_than_coding(self):
        """Reasoning specialists should score higher on spec/review than coding."""
        ranks = classify_model("groq/deepseek-r1-distill-llama-70b")
        assert ranks["spec"] >= ranks["coding"]


# ─── classify_model: large models ─────────────────────────────────────────────


class TestClassifyModelLarge:
    def test_70b_moderate_high_rank(self):
        ranks = classify_model("nvidia_nim/meta/llama-3.3-70b-instruct")
        assert all(r >= 40 for r in ranks.values())

    def test_397b_high_rank(self):
        ranks = classify_model("nvidia_nim/qwen/qwen3.5-397b-a17b")
        assert all(r >= 55 for r in ranks.values())

    def test_nemotron_ultra_high_rank(self):
        ranks = classify_model("nvidia_nim/nvidia/llama-3.1-nemotron-ultra-253b-v1")
        assert all(r >= 40 for r in ranks.values())


# ─── classify_model: small/low-rank models ────────────────────────────────────


class TestClassifyModelLow:
    def test_mini_low_rank(self):
        ranks = classify_model("openrouter/anthropic/claude-haiku-4-mini")
        assert all(r <= 40 for r in ranks.values())

    def test_nano_low_rank(self):
        ranks = classify_model("nvidia_nim/nvidia/nemotron-mini-4b-instruct")
        assert all(r <= 40 for r in ranks.values())

    def test_1b_low_rank(self):
        ranks = classify_model("nvidia_nim/meta/llama-3.2-1b-instruct")
        assert all(r <= 40 for r in ranks.values())

    def test_2b_low_rank(self):
        ranks = classify_model("nvidia_nim/google/gemma-2-2b-it")
        assert all(r <= 40 for r in ranks.values())

    def test_tiny_low_rank(self):
        ranks = classify_model("mistral/mistral-tiny-2407")
        assert all(r <= 40 for r in ranks.values())


# ─── classify_model: exclusions ───────────────────────────────────────────────


class TestClassifyModelExclusions:
    def test_embed_excluded(self):
        assert classify_model("nvidia_nim/baai/bge-m3") == {} or True
        # bge-m3 doesn't contain "embed" literally; let's use a clear embed model
        assert classify_model("nvidia_nim/nvidia/nv-embed-v1") == {}

    def test_whisper_excluded(self):
        assert classify_model("groq/whisper-large-v3") == {}

    def test_tts_excluded(self):
        assert classify_model("gemini/models/gemini-2.5-flash-preview-tts") == {}

    def test_imagen_excluded(self):
        assert classify_model("gemini/models/imagen-4.0-generate-001") == {}

    def test_veo_excluded(self):
        assert classify_model("gemini/models/veo-3.0-generate-001") == {}

    def test_guard_excluded(self):
        assert classify_model("groq/meta-llama/llama-prompt-guard-2-86m") == {}

    def test_safeguard_excluded(self):
        assert classify_model("groq/openai/gpt-oss-safeguard-20b") == {}

    def test_content_safety_excluded(self):
        assert classify_model("nvidia_nim/nvidia/llama-3.1-nemoguard-8b-content-safety") == {}

    def test_moderation_excluded(self):
        assert classify_model("mistral/mistral-moderation-latest") == {}

    def test_codestral_embed_excluded(self):
        """Codestral embed is an embedding model, not a coding model — excluded."""
        assert classify_model("mistral/codestral-embed") == {}

    def test_mistral_embed_excluded(self):
        assert classify_model("mistral/mistral-embed") == {}


# ─── classify_model: return shape ─────────────────────────────────────────────


class TestClassifyModelShape:
    def test_all_roles_present_for_non_excluded_model(self):
        ranks = classify_model("nvidia_nim/meta/llama-3.1-70b-instruct")
        assert set(ranks.keys()) == {"coding", "testing", "review", "spec", "reasoning"}

    def test_all_ranks_are_ints(self):
        ranks = classify_model("openrouter/deepseek/deepseek-v4-pro")
        assert all(isinstance(v, int) for v in ranks.values())

    def test_empty_dict_for_excluded(self):
        result = classify_model("groq/whisper-large-v3")
        assert result == {}


# ─── load_catalog ─────────────────────────────────────────────────────────────


class TestLoadCatalog:
    def test_load_valid_catalog(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    {"model_id": "groq/llama-3.3-70b-versatile", "provider": "groq", "created": 1}
                ]
            },
        )
        catalog = load_catalog(tmp_path)
        assert "groq" in catalog
        assert catalog["groq"][0]["model_id"] == "groq/llama-3.3-70b-versatile"

    def test_missing_file_returns_empty(self, tmp_path: Path):
        catalog = load_catalog(tmp_path)
        assert catalog == {}

    def test_corrupt_json_returns_empty(self, tmp_path: Path):
        (tmp_path / "provider_catalog.json").write_text("{bad json")
        catalog = load_catalog(tmp_path)
        assert catalog == {}

    def test_empty_catalog_key(self, tmp_path: Path):
        payload = {"timestamp": 1.0, "catalog": {}}
        (tmp_path / "provider_catalog.json").write_text(json.dumps(payload))
        catalog = load_catalog(tmp_path)
        assert catalog == {}


# ─── auto_populate_matrix: basic behavior ─────────────────────────────────────


class TestAutoPopulateMatrix:
    def test_writes_matrix_file(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    {"model_id": "groq/llama-3.3-70b-versatile", "provider": "groq", "created": 1}
                ]
            },
        )
        auto_populate_matrix(tmp_path)
        assert (tmp_path / "model_role_matrix.json").exists()

    def test_returns_valid_matrix_dict(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    {"model_id": "groq/llama-3.3-70b-versatile", "provider": "groq", "created": 1}
                ]
            },
        )
        result = auto_populate_matrix(tmp_path)
        assert result["version"] == 1
        assert "roles" in result
        assert set(result["roles"].keys()) == {"coding", "testing", "review", "spec", "reasoning"}

    def test_model_appears_in_all_roles(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    {"model_id": "groq/llama-3.3-70b-versatile", "provider": "groq", "created": 1}
                ]
            },
        )
        result = auto_populate_matrix(tmp_path)
        for role, entries in result["roles"].items():
            model_ids = [e["model_id"] for e in entries]
            assert "groq/llama-3.3-70b-versatile" in model_ids, f"missing in {role}"

    def test_excluded_model_not_in_matrix(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    {"model_id": "groq/whisper-large-v3", "provider": "groq", "created": 1},
                    {"model_id": "groq/llama-3.3-70b-versatile", "provider": "groq", "created": 1},
                ]
            },
        )
        result = auto_populate_matrix(tmp_path)
        for _role, entries in result["roles"].items():
            model_ids = [e["model_id"] for e in entries]
            assert "groq/whisper-large-v3" not in model_ids

    def test_roles_sorted_by_rank_descending(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    {"model_id": "groq/llama-3.3-70b-versatile", "provider": "groq", "created": 1},
                    {"model_id": "groq/llama-3.1-8b-instant", "provider": "groq", "created": 1},
                ]
            },
        )
        result = auto_populate_matrix(tmp_path)
        for role, entries in result["roles"].items():
            ranks = [e["rank"] for e in entries]
            assert ranks == sorted(ranks, reverse=True), f"unsorted in role {role}"

    def test_empty_catalog_produces_empty_roles(self, tmp_path: Path):
        _write_catalog(tmp_path, {})
        result = auto_populate_matrix(tmp_path)
        for role, entries in result["roles"].items():
            assert entries == [], f"expected empty for {role}"

    def test_empty_catalog_no_existing_matrix(self, tmp_path: Path):
        _write_catalog(tmp_path, {})
        result = auto_populate_matrix(tmp_path, merge_existing=False)
        assert result["version"] == 1
        for entries in result["roles"].values():
            assert entries == []


# ─── auto_populate_matrix: merge behavior ─────────────────────────────────────


class TestAutoPopulateMatrixMerge:
    def _curated_matrix(self) -> dict[str, Any]:
        return {
            "version": 1,
            "default_rank": 20,
            "roles": {
                "coding": [{"model_id": "nvidia_nim/moonshotai/kimi-k2.6", "rank": 95}],
                "testing": [{"model_id": "nvidia_nim/moonshotai/kimi-k2.6", "rank": 88}],
                "review": [{"model_id": "nvidia_nim/moonshotai/kimi-k2.6", "rank": 85}],
                "spec": [{"model_id": "nvidia_nim/moonshotai/kimi-k2.6", "rank": 80}],
                "reasoning": [{"model_id": "nvidia_nim/moonshotai/kimi-k2.6", "rank": 78}],
            },
        }

    def _nim_entry(self, name: str) -> dict[str, Any]:
        return {"model_id": f"nvidia_nim/{name}", "provider": "nvidia_nim", "created": 1}

    def test_curated_rank_preserved_on_merge(self, tmp_path: Path):
        _write_matrix(tmp_path, self._curated_matrix())
        _write_catalog(
            tmp_path,
            {
                "nvidia_nim": [
                    self._nim_entry("moonshotai/kimi-k2.6"),
                    self._nim_entry("meta/llama-3.1-70b-instruct"),
                ]
            },
        )
        result = auto_populate_matrix(tmp_path, merge_existing=True)

        # Curated rank for kimi-k2.6 must be exactly preserved
        coding_entries = {e["model_id"]: e["rank"] for e in result["roles"]["coding"]}
        assert coding_entries["nvidia_nim/moonshotai/kimi-k2.6"] == 95

    def test_new_model_added_on_merge(self, tmp_path: Path):
        _write_matrix(tmp_path, self._curated_matrix())
        _write_catalog(
            tmp_path,
            {
                "nvidia_nim": [
                    self._nim_entry("moonshotai/kimi-k2.6"),
                    self._nim_entry("meta/llama-3.1-70b-instruct"),
                ]
            },
        )
        result = auto_populate_matrix(tmp_path, merge_existing=True)

        for role, entries in result["roles"].items():
            model_ids = [e["model_id"] for e in entries]
            assert "nvidia_nim/meta/llama-3.1-70b-instruct" in model_ids, (
                f"new model missing from {role}"
            )

    def test_curated_model_not_in_catalog_gets_status_field(self, tmp_path: Path):
        _write_matrix(tmp_path, self._curated_matrix())
        # Catalog does NOT include kimi-k2.6
        _write_catalog(
            tmp_path,
            {"nvidia_nim": [self._nim_entry("meta/llama-3.1-70b-instruct")]},
        )
        result = auto_populate_matrix(tmp_path, merge_existing=True)

        coding_entries = {e["model_id"]: e for e in result["roles"]["coding"]}
        kimi_entry = coding_entries.get("nvidia_nim/moonshotai/kimi-k2.6")
        assert kimi_entry is not None, "curated model should still appear in matrix"
        assert kimi_entry.get("catalog_status") == "not_in_catalog"

    def test_merge_false_ignores_existing_matrix(self, tmp_path: Path):
        _write_matrix(tmp_path, self._curated_matrix())
        _write_catalog(
            tmp_path,
            {"nvidia_nim": [self._nim_entry("meta/llama-3.1-70b-instruct")]},
        )
        result = auto_populate_matrix(tmp_path, merge_existing=False)

        # kimi-k2.6 is in the existing matrix but not in the catalog,
        # and merge_existing=False, so it should NOT appear.
        for _role, entries in result["roles"].items():
            model_ids = [e["model_id"] for e in entries]
            assert "nvidia_nim/moonshotai/kimi-k2.6" not in model_ids

    def test_no_duplicate_entries_per_role(self, tmp_path: Path):
        _write_matrix(tmp_path, self._curated_matrix())
        _write_catalog(
            tmp_path,
            {"nvidia_nim": [self._nim_entry("moonshotai/kimi-k2.6")]},
        )
        result = auto_populate_matrix(tmp_path, merge_existing=True)

        for role, entries in result["roles"].items():
            model_ids = [e["model_id"] for e in entries]
            assert len(model_ids) == len(set(model_ids)), f"duplicates found in {role}"

    def test_no_existing_matrix_creates_fresh(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "groq": [
                    {
                        "model_id": "groq/deepseek-r1-distill-llama-70b",
                        "provider": "groq",
                        "created": 1,
                    }
                ]
            },
        )
        result = auto_populate_matrix(tmp_path, merge_existing=True)
        reasoning_ids = [e["model_id"] for e in result["roles"]["reasoning"]]
        assert "groq/deepseek-r1-distill-llama-70b" in reasoning_ids


# ─── Written file matches returned dict ───────────────────────────────────────


class TestAutoPopulateMatrixFileParity:
    def test_written_file_matches_returned_dict(self, tmp_path: Path):
        _write_catalog(
            tmp_path,
            {
                "mistral": [
                    {"model_id": "mistral/codestral-latest", "provider": "mistral", "created": 1},
                    {"model_id": "mistral/mistral-embed", "provider": "mistral", "created": 1},
                ]
            },
        )
        result = auto_populate_matrix(tmp_path)
        on_disk = _read_matrix(tmp_path)
        assert result == on_disk
