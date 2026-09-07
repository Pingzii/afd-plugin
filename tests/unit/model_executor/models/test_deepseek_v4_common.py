import pytest

pytest.importorskip("vllm")

from afd_plugin.model_executor.models.deepseek_v4_common import (
    _checkpoint_weight_roles,
)


def test_v4_common_weight_policy_supports_gpu_and_ascend_names():
    assert _checkpoint_weight_roles("model.layers.0.ffn.gate.weight") == frozenset(
        ("ffn",),
    )
    assert _checkpoint_weight_roles(
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ) == frozenset(("ffn",))
    assert _checkpoint_weight_roles(
        "model.layers.0.self_attn.q_proj.weight",
    ) == frozenset(("attention",))
    assert _checkpoint_weight_roles("model.hc_head_fn") == frozenset(
        ("attention",),
    )
    assert _checkpoint_weight_roles("model.embed_tokens.weight") == frozenset(
        ("attention", "ffn"),
    )
