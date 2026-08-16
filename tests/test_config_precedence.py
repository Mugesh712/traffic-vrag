"""Config precedence: env vars > configs/pipeline.yaml > declared defaults.

Regression tests for a real bug found while wiring M16's ablation switches:
get_settings() used to pass the YAML dict as init kwargs, and pydantic-settings
ranks init kwargs ABOVE environment variables -- so every PIPELINE__* override
was silently ignored, including the Neo4j password override the config file
documents as the way to keep the password out of the repo.

get_settings is lru_cached, so each test clears the cache before reading.
"""
from __future__ import annotations

import pytest

from src.utils.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_yaml_overrides_declared_defaults():
    """The declared default is yolo11m.pt only because the YAML says so; this
    asserts the YAML source is actually being read at all."""
    settings = get_settings()
    assert settings.detect.model_path == "yolo11m.pt"


def test_env_var_overrides_yaml(monkeypatch):
    monkeypatch.setenv("PIPELINE__DETECT__CONF_THRESHOLD", "0.9")
    assert get_settings().detect.conf_threshold == pytest.approx(0.9)


def test_env_var_overrides_yaml_for_a_boolean(monkeypatch):
    """The M16 ablation switches are booleans, and are driven this way."""
    assert get_settings().association.enable_appearance_gate is True
    get_settings.cache_clear()
    monkeypatch.setenv("PIPELINE__ASSOCIATION__ENABLE_APPEARANCE_GATE", "false")
    assert get_settings().association.enable_appearance_gate is False


def test_secret_override_works_as_documented(monkeypatch):
    """configs/pipeline.yaml tells the user to keep the password out of the
    repo with this variable. It has to actually work."""
    monkeypatch.setenv("PIPELINE__NEO4J__PASSWORD", "not-the-committed-one")
    assert get_settings().neo4j.password == "not-the-committed-one"


def test_overriding_one_key_preserves_its_siblings(monkeypatch):
    """A nested override must merge into the YAML section, not replace it --
    otherwise setting one detector field would silently reset the others to
    their declared defaults."""
    monkeypatch.setenv("PIPELINE__DETECT__CONF_THRESHOLD", "0.9")
    settings = get_settings()
    assert settings.detect.conf_threshold == pytest.approx(0.9)
    assert settings.detect.model_path == "yolo11m.pt"  # from YAML, not clobbered
    assert settings.detect.iou_threshold == pytest.approx(0.45)


def test_unset_env_leaves_yaml_values_intact():
    settings = get_settings()
    assert settings.neo4j.password == "traffic-vrag"
    assert settings.association.enable_appearance_gate is True
