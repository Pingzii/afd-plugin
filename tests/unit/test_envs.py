from __future__ import annotations

from afd_plugin.envs import (
    AFD_FORCE_BALANCED_TOPK_IDS,
    AFD_NPU_GRAPH_DIAGNOSTICS,
    force_balanced_topk_ids_enabled,
    npu_graph_diagnostics_enabled,
)


def test_force_balanced_topk_ids_env_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(AFD_FORCE_BALANCED_TOPK_IDS, raising=False)

    assert force_balanced_topk_ids_enabled() is False


def test_force_balanced_topk_ids_env_accepts_true_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(AFD_FORCE_BALANCED_TOPK_IDS, value)

        assert force_balanced_topk_ids_enabled() is True


def test_npu_graph_diagnostics_env_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(AFD_NPU_GRAPH_DIAGNOSTICS, raising=False)

    assert npu_graph_diagnostics_enabled() is False


def test_npu_graph_diagnostics_env_accepts_true_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv(AFD_NPU_GRAPH_DIAGNOSTICS, value)

        assert npu_graph_diagnostics_enabled() is True
