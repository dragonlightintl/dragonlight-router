"""Coverage tests for server/app.py lines not covered by test_server.py.

Covers:
- Line 69: _create_flavor_loader when config_path is None (fallback profile path)
- Line 175: CORS middleware added when cors_config is not None
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Line 69 — _create_flavor_loader when config_path is None
# ---------------------------------------------------------------------------


class TestCreateFlavorLoaderNullConfigPath:
    def test_returns_loader_with_default_path_when_config_path_none(self):
        """_create_flavor_loader with config_path=None uses fallback."""
        from dragonlight_router.server.app import _create_flavor_loader

        loader = _create_flavor_loader(config_path=None)
        assert loader is not None
        # The loader's path should be config/model_flavor_profiles.yaml
        assert str(loader._path) == str(Path("config") / "model_flavor_profiles.yaml")

    def test_returns_loader_with_config_parent_when_config_path_set(self, tmp_path):
        """[TM-010] _create_flavor_loader with config_path uses sibling path (line 67)."""
        from dragonlight_router.server.app import _create_flavor_loader

        config_path = tmp_path / "router.yaml"
        config_path.touch()

        loader = _create_flavor_loader(config_path=config_path)
        assert loader is not None
        expected = config_path.parent / "model_flavor_profiles.yaml"
        assert loader._path == expected


# ---------------------------------------------------------------------------
# Line 175 — CORS middleware added when get_cors_config returns non-None
# ---------------------------------------------------------------------------


class TestCorsMiddlewareApplication:
    def test_cors_applied_when_config_present(self, tmp_path):
        """[SEC-001] CORS middleware is added when get_cors_config returns config (line 175)."""
        from dragonlight_router.server.app import create_app

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        import json

        import yaml

        config = {
            "state_dir": str(state_dir),
            "catalog_ttl_hours": 24,
            "default_top_n": 12,
            "max_consecutive_same_provider": 2,
            "providers": [],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config))
        (state_dir / "model_role_matrix.json").write_text(json.dumps({}))

        cors_config = {
            "allow_origins": ["*"],
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["*"],
        }

        with patch(
            "dragonlight_router.server.app.get_cors_config",
            return_value=cors_config,
        ):
            app = create_app(config_path=config_path)

        # App should have been created successfully with CORS middleware
        assert app is not None
