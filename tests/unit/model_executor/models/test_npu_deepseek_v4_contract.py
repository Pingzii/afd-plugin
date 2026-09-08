from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4 as router,
)
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v4_p2p as adapter,
)

native = adapter.native


class _FakeConnector:
    def __init__(self) -> None:
        self.sent = None

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.sent = (hidden_states, context, kwargs)

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        return ref_tensor * 0.5


class _RecordingDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = 0
        self.input_ids = None

    def forward(
        self,
        positions,
        hidden_states,
        residual,
        llama_4_scaling=None,
        input_ids=None,
    ):
        self.input_ids = input_ids
        return hidden_states, residual


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


@pytest.mark.parametrize(
    ("connector", "expected_model_cls", "requires_input_ids"),
    [
        ("CAMAsyncAFDConnector", router.AFDDeepseekV4Model, False),
        ("CAMP2pAFDConnector", adapter.AFDNPUDeepseekV4Model, True),
    ],
)
def test_npu_v4_router_selects_model_for_connector(
    monkeypatch,
    connector,
    expected_model_cls,
    requires_input_ids,
):
    afd_config = SimpleNamespace(connector=connector)
    monkeypatch.setattr(
        router,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )

    selected = {}

    def fake_async_init(self, *, vllm_config, prefix=""):
        selected["model_cls"] = self.model_cls
        selected["requires_input_ids"] = self.afd_requires_input_ids

    monkeypatch.setattr(
        router.AFDDeepseekV4ForCausalLM,
        "__init__",
        fake_async_init,
    )

    router.AFDNPUDeepseekV4ForCausalLM(vllm_config=SimpleNamespace())

    assert selected == {
        "model_cls": expected_model_cls,
        "requires_input_ids": requires_input_ids,
    }


def test_npu_v4_native_import_supports_flat_module(monkeypatch):
    flat_module = ModuleType("vllm_ascend.models.deepseek_v4")
    monkeypatch.setattr(adapter, "import_module", lambda _name: flat_module)

    assert adapter._import_native_deepseek_v4() is flat_module


def test_npu_v4_native_import_supports_package_module(monkeypatch):
    package_module = ModuleType("vllm_ascend.models.deepseek_v4")
    package_module.__path__ = []
    model_module = ModuleType("vllm_ascend.models.deepseek_v4.model")
    imported_names = []

    def fake_import_module(name):
        imported_names.append(name)
        return model_module if name.endswith(".model") else package_module

    monkeypatch.setattr(adapter, "import_module", fake_import_module)

    assert adapter._import_native_deepseek_v4() is model_module
    assert imported_names == [
        "vllm_ascend.models.deepseek_v4",
        "vllm_ascend.models.deepseek_v4.model",
    ]


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


def test_npu_v4_model_forward_preserves_input_ids_without_mtp(monkeypatch):
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(adapter.native, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        adapter.AFDNPUDeepseekV4Model,
        "hc_head",
        lambda _self, hidden_states, *_args: hidden_states[:, 0, :],
    )

    model = adapter.AFDNPUDeepseekV4Model.__new__(
        adapter.AFDNPUDeepseekV4Model,
    )
    torch.nn.Module.__init__(model)
    layer = _RecordingDecoderLayer()
    model.hc_mult = 1
    model.start_layer = 0
    model.end_layer = 1
    model.layers = torch.nn.ModuleList([layer])
    model.aux_hidden_state_layers = ()
    model._mtp_hidden_buffer = None
    model.hc_head_fn = None
    model.hc_head_scale = None
    model.hc_head_base = None
    model.norm = torch.nn.Identity()

    input_ids = torch.tensor([7, 11], dtype=torch.int32)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    inputs_embeds = torch.ones((2, 4), dtype=torch.float16)

    output = adapter.AFDNPUDeepseekV4Model.forward(
        model,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    assert layer.input_ids is input_ids
    assert torch.equal(output, inputs_embeds)


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


def test_npu_v4_model_preserves_eager_compile_contract(monkeypatch):
    afd_config = SimpleNamespace(
        compute_gate_on_attention=False,
        connector="CAMP2pAFDConnector",
        role="ffn",
    )
    compilation_config = SimpleNamespace(mode="none")
    hf_config = SimpleNamespace(
        hc_eps=1e-5,
        hc_mult=1,
        hidden_size=8,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        vocab_size=16,
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation_config,
        lora_config=None,
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_config=hf_config,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
            enable_eplb=False,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
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
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    monkeypatch.setattr(
        adapter.native,
        "make_layers",
        lambda *_args, **_kwargs: (0, 0, torch.nn.ModuleList()),
    )
    monkeypatch.setattr(
        adapter.native,
        "make_pp_empty_intermediate_tensors",
        lambda _model, factory: factory,
        raising=False,
    )
    monkeypatch.setattr(
        adapter.native,
        "PPMissingLayer",
        torch.nn.Identity,
    )
    monkeypatch.setattr(
        adapter.AFDNPUDeepseekV4Model,
        "forward",
        lambda _self: "eager-forward",
    )

    model = adapter.AFDNPUDeepseekV4Model(vllm_config=vllm_config)

    assert model.vllm_config is vllm_config
    assert model.compilation_config is compilation_config
    assert model.do_not_compile is True
    assert model() == "eager-forward"
