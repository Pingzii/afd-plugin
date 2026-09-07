from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm_ascend.models.deepseek_v4 import model as native  # noqa: E402

from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402


class _FakeConnector:
    def __init__(self) -> None:
        self.sent = None

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.sent = (hidden_states, context, kwargs)

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        return ref_tensor * 0.5


def test_npu_v4_wrapper_uses_ascend_native_classes():
    assert issubclass(
        adapter.AFDNPUDeepseekV4DecoderLayer,
        native.DeepseekV2DecoderLayer,
    )
    assert issubclass(
        adapter.AFDNPUDeepseekV4ForCausalLM,
        native.AscendDeepseekV4ForCausalLM,
    )
    registered_model_cls = adapter.AFDNPUDeepseekV4ForCausalLM.model_cls
    assert registered_model_cls is adapter.AFDNPUDeepseekV4Model
    assert adapter.AFDNPUDeepseekV4ForCausalLM.afd_requires_input_ids


def test_npu_v4_remote_ffn_sends_input_ids(monkeypatch):
    connector = _FakeConnector()
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(ubatch_idx=1),
    )
    monkeypatch.setattr(
        adapter,
        "maybe_apply_dbo_yield",
        lambda hidden_states, *, role: hidden_states,
    )
    proxy = adapter.RemoteNPUDeepseekV4FFN(layer_idx=3)
    hidden_states = torch.ones((2, 4), dtype=torch.float16)
    input_ids = torch.tensor([7, 11], dtype=torch.int32)

    output = proxy(hidden_states, input_ids)

    assert connector.sent is not None
    assert connector.sent[1].metadata.layer_idx == 3
    assert connector.sent[1].metadata.stage_idx == 1
    assert connector.sent[2]["input_ids"] is input_ids
    assert torch.equal(output, hidden_states * 0.5)


def test_npu_v4_model_requires_eager_a5_p2p(monkeypatch):
    afd_config = SimpleNamespace(
        compute_gate_on_attention=False,
        connector="CAMP2pAFDConnector",
        role="attention",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
            enable_eplb=False,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        lora_config=None,
        speculative_config=None,
    )
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="npu"),
    )
    monkeypatch.setattr(adapter, "is_a5", lambda: True)
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )

    with pytest.raises(RuntimeError, match="requires --enforce-eager"):
        adapter.AFDNPUDeepseekV4Model(vllm_config=vllm_config)
