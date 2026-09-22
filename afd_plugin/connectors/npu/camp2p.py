# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Synchronous CAMP2p connector for Attention-FFN Disaggregation on NPU.

``CAMP2pAFDConnector`` exchanges hidden states and FFN outputs through Ascend
CAMP2p custom operators backed by HCCL. A separate Gloo group carries DP
metadata from Attention to FFN so each FFN rank can determine the tensor size
for its mapped Attention ranks.

The data path supports eager execution and ``FULL_DECODE_ONLY`` ACL graphs.

See ``docs/npu/CAM_P2P_CONNECTOR_USER_GUIDE.md`` for configuration and launch
examples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.compat.npu import ensure_cam_p2p_ops_available
from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_int,
    coerce_extra_positive_int,
    coerce_optional_extra_positive_int,
)
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import (
    create_hccl_process_group_options,
    init_afd_process_group,
    topology_from_config,
)
from afd_plugin.hash_token_ids import validate_hash_token_ids

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_CAMP2P_CUSTOM_OPS_REGISTERED = False
logger = init_logger(__name__)

# Padding value for token ids that only exist to fill a padded transfer. The
# receiving FFN maps it to token 0 before routing, so a padded row can never
# index outside the token-to-expert table.
_PAD_HASH_TOKEN_ID: Final[int] = -1

_CAMP2P_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "core_num",
        "attn_core_num",
        "ffn_core_num",
        "compute_gate_on_attention",
        "hccl_buffer_size",
        "quant_mode",
    },
)


@dataclass(frozen=True)
class CAMP2PExtraInfo(ConnectorExtraInfo):
    """Typed CAMP2P connector configuration.

    Attributes:
        core_num: Default number of AIV cores used by each AFD role.
        attn_core_num: Optional Attention-role override for ``core_num``.
        ffn_core_num: Optional FFN-role override for ``core_num``.
        compute_gate_on_attention: Whether Attention computes MoE gate outputs.
        hccl_buffer_size: Optional buffer size in MB for CAMP2P HCCL domains.
        quant_mode: CAM quantization mode; the current runtime supports only 0.
    """

    core_num: int = 8
    attn_core_num: int | None = None
    ffn_core_num: int | None = None
    compute_gate_on_attention: bool = False
    hccl_buffer_size: int | None = None
    quant_mode: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> CAMP2PExtraInfo:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__} connector_extra_config must be a mapping, "
                f"got {type(raw).__name__}",
            )
        unknown = sorted(
            str(key) for key in raw if key not in _CAMP2P_EXTRA_CONFIG_FIELDS
        )
        if unknown:
            raise ValueError(
                "unknown CAMP2P connector_extra_config field(s): " + ", ".join(unknown),
            )

        return cls(
            core_num=coerce_extra_positive_int(
                raw.get("core_num", 8),
                field_name="core_num",
            ),
            attn_core_num=coerce_optional_extra_positive_int(
                raw.get("attn_core_num"),
                field_name="attn_core_num",
            ),
            ffn_core_num=coerce_optional_extra_positive_int(
                raw.get("ffn_core_num"),
                field_name="ffn_core_num",
            ),
            compute_gate_on_attention=coerce_extra_bool(
                raw.get("compute_gate_on_attention", False),
                field_name="compute_gate_on_attention",
            ),
            hccl_buffer_size=coerce_optional_extra_positive_int(
                raw.get("hccl_buffer_size"),
                field_name="hccl_buffer_size",
            ),
            quant_mode=coerce_extra_int(
                raw.get("quant_mode", 0),
                field_name="quant_mode",
            ),
        )

    def aiv_num_for_role(self, role: str) -> int:
        if role == "attention" and self.attn_core_num is not None:
            return self.attn_core_num
        if role == "ffn" and self.ffn_core_num is not None:
            return self.ffn_core_num
        return self.core_num

    def validate_supported(self) -> None:
        if self.compute_gate_on_attention:
            raise RuntimeError(
                "AFD NPU runtime does not support compute_gate_on_attention=true yet",
            )
        if self.quant_mode != 0:
            raise RuntimeError("AFD NPU runtime currently supports only quant_mode=0")

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "core_num": self.core_num,
            "compute_gate_on_attention": self.compute_gate_on_attention,
            "quant_mode": self.quant_mode,
        }
        if self.attn_core_num is not None:
            result["attn_core_num"] = self.attn_core_num
        if self.ffn_core_num is not None:
            result["ffn_core_num"] = self.ffn_core_num
        if self.hccl_buffer_size is not None:
            result["hccl_buffer_size"] = self.hccl_buffer_size
        return result


@dataclass(slots=True)
class CAMP2PTransferState(AFDTransferState):
    """CAMP2P payload metadata carried between recv and send phases.

    This class stores what CAMP2p reads back itself while data travels from
    Attention to FFN and then back to Attention. ``aiv_num``, ``batch_size``,
    ``h`` and ``k`` size the CAMP2p operators, and ``atten_batch_size`` saves the
    A2E-returned Attention token count that the FFN-to-Attention send requires.
    ``x_active_mask`` and ``cam_p2p_ep_name`` are the A2E-returned active-token
    mask and HCCL endpoint name captured on the receive path. ``attention_rows``
    is set on the Attention side when the payload had to be padded up to the
    reported tile size; the receive then keeps only that many rows, which is what
    the model produced.
    """

    aiv_num: int = 8
    batch_size: int = 0
    h: int = 0
    k: int = 1
    atten_batch_size: torch.Tensor | None = None
    x_active_mask: torch.Tensor | None = None
    cam_p2p_ep_name: str | None = None
    attention_rows: int | None = None


@dataclass(frozen=True, slots=True)
class _CAMP2PTopology:
    """Describe where one connector process sits in the communication groups.

    ``world_rank`` is the process number in the complete AFD group.
    ``p2p_rank`` is its number in the smaller Gloo group used to exchange metadata.
    For an Attention rank, ``dp_metadata_destinations`` lists the FFN ranks
    that should receive its metadata.
    """

    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    dp_metadata_destinations: tuple[int, ...]

    @property
    def p2p_world_size(self) -> int:
        """Return the number of FFN and participating Attention metadata ranks."""
        return self.ffn_size + self.min_size

    @property
    def participates_in_p2p_group(self) -> bool:
        """Return whether this process joins the Gloo DP-metadata group."""
        return self.world_rank < self.ffn_size or self.is_attn_top_min_size_rank

    @property
    def is_attn_top_min_size_rank(self) -> bool:
        """Return whether this is an Attention metadata-sender rank."""
        return self.ffn_size <= self.world_rank < self.ffn_size + self.min_size


def _reported_attention_tokens(forward_context: Any) -> int | None:
    """Return the token count this step's Attention payload has to cover.

    Whenever a step pads its Attention batch, the runner reports that padded
    count for this rank (``v1/worker/attention_model_runner.py`` passes
    ``num_tokens_padded`` into the forward context and reports it to the FFN), and
    the FFN turns the report into the single equal tile A2E reads from each
    Attention peer. A rank that writes fewer rows than it is charged leaves the
    tail of its ids and hidden-state regions unwritten, and the receiving FFN
    reads the neighbouring regions as token ids; a rank that writes more loses
    the extra tokens, because the FFN still reads only its tile.

    Steps split into ubatches report per-stage counts instead, so they keep the
    row count their forward produced and return ``None`` here.
    """

    if getattr(forward_context, "ubatch_slices", None):
        return None
    num_tokens = getattr(forward_context, "num_tokens", None)
    if num_tokens is None:
        return None
    return max(1, int(num_tokens))


def _pad_leading_rows(
    tensor: torch.Tensor,
    pad_rows: int,
    *,
    value: int = 0,
) -> torch.Tensor:
    """Return ``tensor`` with ``pad_rows`` extra rows on its leading dimension."""

    padding = [0, 0] * (tensor.dim() - 1) + [0, pad_rows]
    return torch.nn.functional.pad(tensor, padding, value=value)


def prepare_token_id_transfer(
    input_ids: torch.Tensor,
    *,
    topk: int,
    expected_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack token ids and inert scales for the A2E ids channel.

    The operator transports an ``int32`` ids tensor and a ``float32`` scales
    tensor of shape ``(batch, topk)``. The id repeats across the columns, which
    is what the receiving side collapses back. The scales carry no routing
    weight: this channel moves token identity, not router output.

    Args:
        input_ids: Token-aligned ids for the local Attention tokens.
        topk: Number of routed experts per token; the operator's column count.
        expected_tokens: Token count the transfer metadata declares.

    Returns:
        The ``(ids, scales)`` pair to hand to the operator.

    Raises:
        ValueError: If the ids do not describe exactly ``expected_tokens``
            tokens, which is the alignment invariant for this channel.
    """

    num_tokens = int(input_ids.numel())
    if num_tokens != expected_tokens:
        raise ValueError(
            f"input_ids token count {num_tokens} does not match the AFD "
            f"transfer token count {expected_tokens}",
        )

    ids = (
        input_ids.reshape(-1)
        .to(dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, topk)
        .contiguous()
    )
    scales = torch.zeros(
        (num_tokens, topk),
        dtype=torch.float32,
        device=input_ids.device,
    )
    return ids, scales


def received_token_ids(
    sim_expert_ids: torch.Tensor,
    *,
    expected_tokens: int,
    written_tokens: int,
) -> torch.Tensor:
    """Collapse the A2E ids channel back to a token-aligned id vector.

    Every column of a row repeats that token's id. The operator sizes the output
    to the capacity it was given rather than to the rows it writes, and its host
    inference multiplies that capacity by the number of Attention peers, so the
    shape alone cannot prove which rows arrived. The caller therefore states the
    written row count explicitly.

    Args:
        sim_expert_ids: The operator's ids output.
        expected_tokens: Token count derived from the FFN rank's DP metadata,
            which is what the FFN compute will actually run on.
        written_tokens: Rows the transfer writes, which A2E derives as
            ``peers * (batch_size // peers)`` and never exceeds
            ``expected_tokens`` unless the peers are even.

    Returns:
        A one-dimensional ``int32`` tensor of length ``expected_tokens``.

    Raises:
        ValueError: If the operator's capacity cannot hold the written rows, or
            if fewer rows are written than the FFN rank computes on. The second
            case is the dangerous one: the remaining rows hold uninitialised
            device memory, and a token-keyed router turns each of them into an
            out-of-range table lookup.
    """

    declared_tokens = int(sim_expert_ids.shape[0])
    if declared_tokens < written_tokens:
        raise ValueError(
            f"A2E declared {declared_tokens} id rows but the transfer writes "
            f"{written_tokens}; the ids channel is not sized for this topology",
        )
    if written_tokens < expected_tokens:
        raise ValueError(
            f"received {written_tokens} token ids but the FFN rank computes on "
            f"{expected_tokens} tokens; the last "
            f"{expected_tokens - written_tokens} rows hold uninitialised device "
            "memory, which a token-keyed router reads as an out-of-range id. The "
            "ids channel is not aligned with the FFN token layout",
        )
    # The columns are replicas of the same id, so the first one carries it.
    return sim_expert_ids[:expected_tokens, 0].contiguous()


class CAMP2pAFDConnector(AFDConnectorBase):
    """Move model data between Attention and FFN workers on Ascend NPU.

    The connector owns HCCL process-group setup and CAMP2P custom-op transfers.
    Runtime validation rejects unsupported nonzero quantization modes and
    compute-gate-on-attention settings.

    DP metadata operations do not live on the connector itself: they are
    provided by the pluggable ``CAMP2pAFDControlPlane`` instance created at
    construction time and exposed as ``control_plane``. The connector still
    owns the ``p2p`` process group the control plane transmits over, because
    creating that group is part of the collective ``init_afd_connector``
    ordering.
    """

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> CAMP2PExtraInfo:
        return CAMP2PExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        """Read the configuration and prepare this connector's local state.

        This method calculates the process ranks and reads the model dimensions.
        It does not connect to the other Attention or FFN processes yet;
        :meth:`init_afd_connector` creates those connections later.

        Args:
            rank: Rank provided by the vLLM worker.
            local_rank: NPU device number used by this worker.
            vllm_config: Model, scheduler, and parallel configuration from vLLM.
            afd_config: AFD role, host, port, rank counts, and extra settings.
            role_rank: Runtime rank within the configured AFD role group.
        """
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._initialized = False
        self.topology = build_camp2p_topology(afd_config, role_rank)
        self.world_rank = self.topology.world_rank
        self.p2p_rank = self.topology.p2p_rank
        self.attn_size = self.topology.attention_size
        self.ffn_size = self.topology.ffn_size
        self.min_size = self.topology.min_size
        self.ratio = self.attn_size // self.ffn_size
        self.dst_list = list(self.topology.dp_metadata_destinations)
        self.dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.afd_pg_list: list[ProcessGroup] = []
        self.afd_pg: ProcessGroup | None = None
        self.p2p_pg: ProcessGroup | None = None
        self.ffn_pg: ProcessGroup | None = None
        self.hccl_comm_name = ""
        self.hccl_comm_name2 = ""
        self.hccl_comm_name3 = ""
        self.hccl_comm_name1 = ""
        self.hccl_comm_name_list: list[str] = []
        extra_info = cast(CAMP2PExtraInfo, self.extra_info)
        self.aiv_num = extra_info.aiv_num_for_role(afd_config.role)
        self.hccl_buffer_size_mb = extra_info.hccl_buffer_size
        hf_config = vllm_config.model_config.hf_config
        self.hidden_size = hf_config.hidden_size
        self.num_experts_per_tok = hf_config.num_experts_per_tok
        self.num_routed_experts = hf_config.n_routed_experts
        # Row count of the model's token-to-expert (Hash) tables: token ids the
        # ids channel carries are indices into them.
        self.vocab_size = int(hf_config.vocab_size)
        self.control_plane = CAMP2pAFDControlPlane(self)
        self._receive_buffers: dict[tuple[Any, ...], torch.Tensor] = {}

    @property
    def is_initialized(self) -> bool:
        """Return ``True`` after all CAMP2p connections have been created."""
        return self._initialized

    def init_afd_connector(self) -> None:
        """Connect this process to the other Attention and FFN processes.

        The method creates one HCCL group for each batch or ubatch. These groups
        carry hidden states and FFN results. FFN processes also create a group
        for MoE communication. A smaller Gloo group carries token counts and
        other batch information from Attention to FFN.

        The method returns immediately if initialization already succeeded.
        Otherwise, it may wait until every required process joins.

        Raises:
            RuntimeError: If the CAMP2p operators are unavailable or a
                communication group cannot be created.
        """
        if self._initialized:
            return
        import torch_npu  # noqa: F401

        ensure_cam_p2p_ops_available()

        _register_camp2p_custom_ops()

        num_ubatches = max(1, self.vllm_config.parallel_config.num_ubatches)
        self.afd_pg_list = []
        self.hccl_comm_name_list = []
        for ubatch_idx in range(num_ubatches):
            group_name = "afd" if ubatch_idx == 0 else f"afd{ubatch_idx}"
            afd_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.attn_size,
                rank=self.world_rank,
                group_name=group_name,
                timeout=timedelta(minutes=30),
                pg_options=create_hccl_process_group_options(
                    self.hccl_buffer_size_mb,
                ),
            )
            self.afd_pg_list.append(afd_pg)
            backend = afd_pg._get_backend(torch.device("npu"))
            self.hccl_comm_name_list.append(
                str(backend.get_hccl_comm_name(self.world_rank)),
            )
        self.afd_pg = self.afd_pg_list[0]
        self.hccl_comm_name = self.hccl_comm_name_list[0]
        self.hccl_comm_name2 = (
            self.hccl_comm_name_list[1] if num_ubatches > 1 else self.hccl_comm_name
        )
        self.hccl_comm_name3 = self.hccl_comm_name_list[2] if num_ubatches > 2 else ""

        if self.afd_config.role == "ffn":
            self.ffn_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size,
                rank=self.world_rank,
                group_name="afd_moe",
                timeout=timedelta(minutes=30),
                pg_options=create_hccl_process_group_options(
                    self.hccl_buffer_size_mb,
                ),
            )
            backend = self.ffn_pg._get_backend(torch.device("npu"))
            self.hccl_comm_name1 = str(
                backend.get_hccl_comm_name(self.world_rank),
            )

        if self.topology.participates_in_p2p_group:
            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.topology.p2p_world_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timedelta(minutes=30),
            )

        self._initialized = True

    def close(self) -> None:
        """Close all communication groups created by this connector.

        The method also clears saved HCCL group names and marks the connector as
        uninitialized. It is safe to initialize the connector again afterward.
        """
        groups = [self.p2p_pg, self.ffn_pg, *self.afd_pg_list]
        if self.afd_pg is not None and not self.afd_pg_list:
            groups.append(self.afd_pg)
        destroyed_group_ids: set[int] = set()
        for group in groups:
            if group is not None:
                group_id = id(group)
                if group_id in destroyed_group_ids:
                    continue
                destroyed_group_ids.add(group_id)
                dist.destroy_process_group(group)
        self.p2p_pg = None
        self.ffn_pg = None
        self.afd_pg = None
        self.afd_pg_list = []
        self.hccl_comm_name = ""
        self.hccl_comm_name2 = ""
        self.hccl_comm_name3 = ""
        self.hccl_comm_name1 = ""
        self.hccl_comm_name_list = []
        self._initialized = False

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Send hidden states from an Attention rank to its FFN rank.

        The ubatch number selects the matching HCCL communication group. The
        method saves the values returned by CAMP2p so Attention can later
        receive the FFN result for the same ubatch.

        Args:
            hidden_states: Model data with shape ``(tokens, hidden_size)``.
            context: Transfer context whose ``metadata`` supplies the layer
                number, ubatch number, and token count for this transfer.
            **kwargs: An optional token-aligned ``input_ids`` tensor. When it is
                supplied, the transfer runs with ``compute_gate=1`` so the ids
                reach the FFN rank through the operator's ids channel, and the
                matching ``recv_attn_output(recv_input_ids=True)`` returns them
                on the payload's ``input_ids`` field.

        Raises:
            RuntimeError: If the communication groups are not ready.
            ValueError: If the number of tokens in ``hidden_states`` does not
                match ``context.metadata`` outside a ``torch.compile`` trace, or
                if a supplied ``input_ids`` tensor is malformed.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"CAMP2P metadata token count {metadata.total_tokens}",
            )
        input_ids = cast(torch.Tensor | None, kwargs.get("input_ids"))
        forward_context = get_forward_context()
        reported_rows = _reported_attention_tokens(forward_context)
        attention_rows: int | None = None
        # Branching on the token count specializes the dimension the compiled
        # model declares as dynamic, so the alignment runs outside
        # ``torch.compile`` only, exactly like the shape check above. A compiled
        # or captured step therefore relies on the runner reporting the count it
        # executes, which is what the reported count means.
        if not torch.compiler.is_compiling() and reported_rows is not None:
            model_rows = int(metadata.total_tokens)
            if reported_rows < model_rows:
                raise RuntimeError(
                    f"CAMP2P Attention rank sends {model_rows} tokens but this "
                    f"step reports {reported_rows} tokens per Attention rank "
                    f"(layer={metadata.layer_idx}, ubatch={metadata.stage_idx}). "
                    "A2E reads one equal tile per Attention peer and sends the "
                    "same tile back, so the extra tokens cannot be represented. "
                    "Align the reported token count with the rows the forward "
                    "produces.",
                )
            if reported_rows > model_rows:
                # Pad up to the reported tile: A2E reads exactly that many rows
                # from every peer, so a short payload would otherwise leave the
                # tail of this rank's ids and hidden-state regions unwritten.
                pad_rows = reported_rows - model_rows
                hidden_states = _pad_leading_rows(hidden_states, pad_rows)
                if input_ids is not None:
                    input_ids = _pad_leading_rows(
                        input_ids.reshape(-1),
                        pad_rows,
                        value=_PAD_HASH_TOKEN_ID,
                    )
                metadata = AFDTransferMetadata.create_attention_metadata(
                    layer_idx=metadata.layer_idx,
                    stage_idx=metadata.stage_idx,
                    seq_len=reported_rows,
                )
                attention_rows = model_rows
        expert_ids: torch.Tensor | None = None
        expert_scales: torch.Tensor | None = None
        compute_gate = 0
        if input_ids is not None:
            expert_ids, expert_scales = prepare_token_id_transfer(
                input_ids,
                topk=self.num_experts_per_tok,
                expected_tokens=metadata.total_tokens,
            )
            compute_gate = 1
        transfer_state = CAMP2PTransferState(
            aiv_num=self.aiv_num,
            batch_size=metadata.total_tokens,
            h=self.hidden_size,
            k=self.num_experts_per_tok,
            attention_rows=attention_rows,
        )
        ubatch_idx = metadata.stage_idx
        forward_context.cam_afdtransfer_state = transfer_state
        forward_context.ubatch_idx = ubatch_idx

        torch.ops.vllm.afd_camp2p_send_attn_output(
            hidden_states,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            transfer_state.aiv_num,
            compute_gate,
            expert_ids,
            expert_scales,
        )
        return None

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Receive the processed model data from FFN on an Attention rank.

        Args:
            ref_tensor: Tensor supplying the expected shape and storage for
                the receive operation.
            ubatch_idx: Ubatch to receive. Defaults to ``0``.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            The model data returned by FFN.

        Raises:
            RuntimeError: If communication is not ready or the matching
                Attention send information was lost.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        transfer_state = getattr(get_forward_context(), "cam_afdtransfer_state", None)
        if transfer_state is None:
            raise RuntimeError("CAMP2P Attention side is missing connector data")
        get_forward_context().ubatch_idx = ubatch_idx
        # A padded send makes the FFN return the padded tile, so receive into a
        # tile-sized buffer and hand back only the rows this rank produced.
        destination = ref_tensor
        attention_rows = transfer_state.attention_rows
        if attention_rows is not None:
            destination = self._padded_receive_buffer(
                ref_tensor,
                transfer_state.batch_size,
            )
        received = torch.ops.vllm.afd_camp2p_recv_ffn_output(
            destination,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            transfer_state.aiv_num,
        )
        if attention_rows is None:
            return received
        return received[:attention_rows]

    def _padded_receive_buffer(
        self,
        ref_tensor: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        """Return a reusable ``rows``-row receive buffer shaped like ``ref_tensor``.

        The buffer is cached per shape so a FULL graph capture and its replays
        reuse one allocation instead of allocating inside the captured region.
        """

        key = (rows, *ref_tensor.shape[1:], ref_tensor.dtype)
        buffer = self._receive_buffers.get(key)
        if buffer is None:
            buffer = torch.empty(
                (rows, *ref_tensor.shape[1:]),
                dtype=ref_tensor.dtype,
                device=ref_tensor.device,
            )
            self._receive_buffers[key] = buffer
        return buffer

    def recv_attn_output(
        self, ubatch_idx: int = 0, **kwargs: Any
    ) -> AFDA2FTransferPayload:
        """Receive hidden states from Attention on an FFN rank.

        The ubatch number selects the expected token count and HCCL group.
        Values returned by CAMP2p are saved so the FFN result can be sent back
        to the correct Attention ranks.

        Args:
            ubatch_idx: Ubatch number, starting from ``0``.
            **kwargs: May provide existing transfer information or the layer
                number needed to create it. ``recv_input_ids`` states that this
                FFN rank expects token ids on the transfer, which selects the
                operator's ids mode and makes the connector validate and expose
                the operator's ids slot on the payload. The sending rank has to
                select the same mode, so only request ids for a run whose
                Attention role transports them.

        Returns:
            The received hidden states, the information FFN needs to process them
            and send the result back, and the transported ``input_ids`` when the
            ids mode was selected.

        Raises:
            RuntimeError: If communication is not ready, transfer information
                is missing, the requested ubatch group does not exist, or the
                Attention peers of this FFN rank have uneven token counts, which
                A2E's equal-tile layout cannot represent.
            ValueError: If ids were requested but do not align with the FFN
                rank's token layout.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        layer_idx: int = kwargs.get("layer_idx", 0)
        max_num_tokens: int = kwargs.get("max_num_tokens", 0)
        recv_input_ids: bool = bool(kwargs.get("recv_input_ids", False))
        compute_gate_mode = 1 if recv_input_ids else 0
        batch_size = _num_tokens_for_ffn_rank(
            self.dp_metadata_list,
            ubatch_idx,
            ffn_rank=self.role_rank,
            attention_size=self.attn_size,
            ffn_size=self.ffn_size,
            fallback=max_num_tokens,
        )
        # A2E lays this rank's ids and hidden states out as one tile per
        # Attention peer, so the group's counts have to be even before the
        # transfer is issued rather than after the device faults or the FFN
        # computes on a shifted row order.
        attention_group = _attention_group_token_counts(
            self.dp_metadata_list.get(ubatch_idx),
            ffn_rank=self.role_rank,
            attention_size=self.attn_size,
            ffn_size=self.ffn_size,
        )
        tiles = 1
        if attention_group:
            _check_a2e_token_layout(
                attention_group,
                ffn_rank=self.role_rank,
                attention_size=self.attn_size,
                ffn_size=self.ffn_size,
            )
            tiles = len(attention_group)
        # A2E writes ``tiles * (batch_size // tiles)`` id rows, which its host
        # shape inference then multiplies by ``tiles``, so the declared capacity
        # cannot prove the tail rows arrived.
        written_tokens = tiles * (batch_size // tiles)
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=[batch_size],
        )
        custom_states = CAMP2PTransferState(
            aiv_num=self.aiv_num,
            batch_size=batch_size,
            h=self.hidden_size,
            k=self.num_experts_per_tok,
        )
        context = AFDTransferContext(
            metadata=metadata,
            states=custom_states,
        )

        group_ep = _get_group_ep(
            ubatch_idx,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
        )
        outputs = torch.ops.afd_ascend.a2e(
            torch.tensor([], dtype=torch.bfloat16, device="npu"),
            torch.tensor([], dtype=torch.int32, device="npu"),
            torch.tensor([], dtype=torch.float32, device="npu"),
            custom_states.batch_size,
            custom_states.h,
            custom_states.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            group_ep,
            custom_states.aiv_num,
            compute_gate_mode,
        )
        custom_states.atten_batch_size = outputs[3]
        custom_states.x_active_mask = outputs[4]
        custom_states.cam_p2p_ep_name = self.hccl_comm_name1
        # The ids slot is only written in the operator's ids mode, so the mode has
        # to match the sending rank's ``compute_gate``. It is declared here by the
        # receiving FFN rank through ``recv_input_ids`` rather than inferred from
        # the returned tensor: a genuine single-token layer would otherwise be
        # indistinguishable from the operator's placeholder. Reading the slot in
        # the other mode would hand the model uninitialised device memory as token
        # ids, which a token-keyed router turns into an out-of-range table read.
        received_ids: torch.Tensor | None = None
        if compute_gate_mode == 1:
            received_ids = received_token_ids(
                outputs[1],
                expected_tokens=batch_size,
                written_tokens=written_tokens,
            )
            validate_hash_token_ids(
                received_ids,
                table_rows=self.vocab_size,
                context="CAMP2P FFN-side Hash routing",
            )
        return AFDA2FTransferPayload(
            hidden_states=outputs[0],
            context=context,
            input_ids=received_ids,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Send processed model data from an FFN rank back to Attention.

        Args:
            ffn_output: Model data produced by the FFN layers.
            context: Transfer context saved when FFN received the Attention
                output; its ``states`` carries the CAMP2p receive-time results.
            **kwargs: An optional ``ubatch_idx``. If omitted, the method uses
                the ubatch number stored in ``context.metadata``.

        Raises:
            RuntimeError: If communication is not ready, required receive
                information is missing, or the ubatch group does not exist.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        states = cast(CAMP2PTransferState, context.states)
        if states.atten_batch_size is None:
            raise RuntimeError("CAMP2P FFN side is missing A2E atten_batch_size")
        ubatch_idx = int(kwargs.get("ubatch_idx", context.metadata.stage_idx))
        group_ep = _get_group_ep(
            ubatch_idx,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
        )
        torch.ops.afd_ascend.e2a(
            ffn_output,
            states.atten_batch_size,
            states.batch_size,
            states.h,
            states.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            group_ep,
            states.aiv_num,
        )
        return None


class CAMP2pAFDControlPlane(AFDControlPlane):
    """DP metadata control plane for ``CAMP2pAFDConnector``.

    Applies DP metadata payloads to the owning connector's state and moves
    them between Attention and FFN ranks over the connector's dedicated
    ``p2p`` gloo process group. The connector creates one instance at
    construction time and exposes it through ``control_plane``; the process
    group itself is created by ``init_afd_connector``.
    """

    def __init__(self, connector: CAMP2pAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        connector.dp_metadata_list = payload.dp_metadata_list
        connector.is_graph_capturing = payload.is_graph_capturing
        connector.is_warmup = payload.is_warmup

    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        if connector.p2p_pg is None:
            return
        if not connector.topology.is_attn_top_min_size_rank:
            return
        # The CAMP2P DP-metadata group runs on gloo, so the wire tensors stay on
        # CPU rather than the NPU device.
        device = torch.device("cpu")
        send_control_payload(
            payload,
            dst=connector.dst_list,
            group=connector.p2p_pg,
            device=device,
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.p2p_pg is None:
            raise RuntimeError("CAMP2P metadata process group is not initialized")
        src = connector.p2p_rank % connector.min_size + connector.ffn_size
        return recv_control_payload(
            src=src,
            group=connector.p2p_pg,
            device=torch.device("cpu"),
        )


def build_camp2p_topology(
    afd_config: AFDConfig,
    role_rank: int,
) -> _CAMP2PTopology:
    """Calculate the communication rank numbers for one process.

    FFN processes come first in the main AFD group, followed by Attention
    processes. All FFN ranks and the first ``min(A, F)`` Attention ranks also
    join the smaller Gloo group that exchanges token counts and batch details.

    Args:
        afd_config: Process role and total Attention/FFN rank counts.
        role_rank: This process's runtime number within its own role.

    Returns:
        This process's rank numbers and metadata destinations.

    Raises:
        ValueError: If the rank counts or this process's role rank are invalid.
    """
    attention_size, ffn_size = topology_from_config(afd_config)
    if attention_size <= 0 or ffn_size <= 0:
        raise ValueError("CAMP2P topology sizes must be positive")
    if attention_size < ffn_size:
        raise ValueError(
            "CAMP2P requires attention_size >= ffn_size, got "
            f"{attention_size} < {ffn_size}",
        )
    if role_rank < 0:
        raise ValueError(f"CAMP2P role rank must be non-negative, got {role_rank}")

    if afd_config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        p2p_rank = role_rank + min(ffn_size, attention_size)
    elif afd_config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        p2p_rank = role_rank
    else:
        raise ValueError(f"unknown AFD role {afd_config.role!r}")

    min_size = min(attention_size, ffn_size)
    destinations: list[int] = []
    if ffn_size <= world_rank < ffn_size + min_size:
        local_attention_rank = world_rank - ffn_size
        dst = local_attention_rank
        while dst < ffn_size:
            destinations.append(dst)
            dst += min_size

    return _CAMP2PTopology(
        role=afd_config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        dp_metadata_destinations=tuple(destinations),
    )


def _attention_token_counts(
    dp_metadata: DPMetadata | AFDDPMetadata | None,
    *,
    attention_size: int,
) -> list[int] | None:
    """Return the per-Attention-rank token counts of one stage.

    ``num_tokens_across_dp_cpu`` holds one count per DP rank, but the Attention
    role also contains the TP workers of each DP rank. Each DP count is
    replicated across its TP workers, which makes the result positional in
    Attention rank order. Returns ``None`` when the metadata cannot describe
    every Attention rank, so callers fall back instead of inventing counts.
    """

    if dp_metadata is None:
        return None
    counts = dp_metadata.num_tokens_across_dp_cpu.flatten().tolist()
    if not counts:
        return None
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        tp_size = attention_size // len(counts)
        counts = [counts[idx // tp_size] for idx in range(attention_size)]
    if len(counts) < attention_size:
        return None
    return [int(count) for count in counts[:attention_size]]


def _attention_group_token_counts(
    dp_metadata: DPMetadata | AFDDPMetadata | None,
    *,
    ffn_rank: int,
    attention_size: int,
    ffn_size: int,
) -> list[int] | None:
    """Return the token counts of the Attention ranks one FFN rank serves.

    Returns ``None`` for a topology this helper does not describe, which leaves
    the caller's fallback path unchanged.
    """

    counts = _attention_token_counts(dp_metadata, attention_size=attention_size)
    if counts is None:
        return None
    if ffn_size <= 0 or attention_size < ffn_size or attention_size % ffn_size != 0:
        return None
    group_size = attention_size // ffn_size
    start_idx = ffn_rank * group_size
    return counts[start_idx : start_idx + group_size]


def _check_a2e_token_layout(
    group_counts: list[int],
    *,
    ffn_rank: int,
    attention_size: int,
    ffn_size: int,
) -> None:
    """Fail fast when A2E cannot lay out this FFN rank's Attention group.

    A2E splits the ids, scales, and hidden-state regions of one FFN rank into
    ``attention_size // ffn_size`` tiles of ``batch_size // tiles`` rows, while
    each Attention rank writes its regions using its own token count
    (``csrc/npu/ascend_kernels/a2e/op_kernel/a2e.h``). The two agree only when
    every Attention peer sends the same number of tokens: one uneven peer shifts
    every later peer's rows and leaves the tail rows of ``simulate_expert_ids``
    unwritten. Those rows hold uninitialised device memory, and a DSV4 Hash layer
    turns each of them into an out-of-range ``tid2eid`` read on device.

    Raises:
        RuntimeError: If the peers of this FFN rank have different token counts.
    """

    tiles = len(group_counts)
    if tiles <= 1 or len(set(group_counts)) == 1:
        return
    raise RuntimeError(
        f"CAMP2P FFN rank {ffn_rank} of {ffn_size} serves {tiles} Attention "
        f"ranks (attention_size={attention_size}) with uneven token counts "
        f"{group_counts}. A2E lays this rank out as {tiles} equal tiles, so "
        "uneven peers misalign every later peer's rows and leave the tail ids "
        "rows unwritten; a token-keyed router then reads uninitialised device "
        "memory as token ids. Make every Attention peer of an FFN rank send the "
        "same token count, or fix the A2E per-sender layout.",
    )


def _num_tokens_for_ffn_rank(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    stage_idx: int,
    *,
    ffn_rank: int,
    attention_size: int,
    ffn_size: int,
    fallback: int,
) -> int:
    """Count the tokens that one FFN rank will receive from Attention.

    An FFN rank may receive data from several consecutive Attention ranks. This
    function adds their token counts. When TP creates several Attention workers
    for one DP rank, it first copies the DP token count to those TP workers.

    Args:
        dp_metadata_list: Token counts received from Attention for each ubatch.
        stage_idx: Ubatch number to inspect.
        ffn_rank: FFN rank whose token count is needed.
        attention_size: Total number of Attention ranks.
        ffn_size: Total number of FFN ranks.
        fallback: Value used when the token counts are missing or incomplete.

    Returns:
        The number of tokens this FFN rank should receive, always at least one.
    """
    group_counts = _attention_group_token_counts(
        dp_metadata_list[stage_idx],
        ffn_rank=ffn_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
    )
    if group_counts is None:
        return max(1, fallback)
    return max(1, sum(group_counts))


def _get_group_ep(
    ubatch_idx: int,
    hccl_comm_name1: str,
    hccl_comm_name2: str,
    hccl_comm_name3: str,
) -> str:
    if ubatch_idx == 1:
        return hccl_comm_name2 or hccl_comm_name1
    if ubatch_idx == 2:
        if not hccl_comm_name3:
            raise RuntimeError("CAMP2P ubatch 2 requires a third HCCL group")
        return hccl_comm_name3
    if ubatch_idx < 0:
        raise RuntimeError(f"CAMP2P ubatch index must be non-negative: {ubatch_idx}")
    return hccl_comm_name1


def _register_camp2p_custom_ops() -> None:
    """Register the CAMP2P send and receive operations once per process.

    The A2E operation sends Attention output to FFN. The E2A operation sends
    the FFN result back to Attention.
    """
    global _CAMP2P_CUSTOM_OPS_REGISTERED
    if _CAMP2P_CUSTOM_OPS_REGISTERED:
        return

    def send_attn_output_impl(
        hidden_states: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
        compute_gate: int,
        expert_ids: torch.Tensor | None,
        expert_scales: torch.Tensor | None,
    ) -> torch.Tensor:
        transfer_state = getattr(get_forward_context(), "cam_afdtransfer_state", None)
        if transfer_state is None:
            transfer_state = CAMP2PTransferState()
        transfer_state.batch_size = batch_size
        transfer_state.h = hidden_size
        transfer_state.k = topk
        transfer_state.aiv_num = aiv_num
        group_ep = _get_group_ep(
            int(getattr(get_forward_context(), "ubatch_idx", 0)),
            hccl_comm_name,
            hccl_comm_name2,
            hccl_comm_name3,
        )

        outputs = torch.ops.afd_ascend.a2e(
            hidden_states,
            expert_ids,
            expert_scales,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            ffn_size,
            attn_size,
            world_rank,
            group_ep,
            transfer_state.aiv_num,
            compute_gate,
        )
        transfer_state.atten_batch_size = outputs[3]
        forward_context = get_forward_context()
        forward_context.cam_afdtransfer_state = transfer_state
        return hidden_states

    def send_attn_output_fake_impl(
        hidden_states: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
        compute_gate: int,
        expert_ids: torch.Tensor | None,
        expert_scales: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the input unchanged while PyTorch inspects the send operation."""
        return hidden_states

    def recv_ffn_output_impl(
        ref_tensor: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
    ) -> torch.Tensor:
        transfer_state = getattr(get_forward_context(), "cam_afdtransfer_state", None)
        if transfer_state is None or transfer_state.atten_batch_size is None:
            raise RuntimeError("CAMP2P Attention side is missing A2E handle data")
        transfer_state.batch_size = batch_size
        transfer_state.h = hidden_size
        transfer_state.k = topk
        transfer_state.aiv_num = aiv_num
        group_ep = _get_group_ep(
            int(getattr(get_forward_context(), "ubatch_idx", 0)),
            hccl_comm_name,
            hccl_comm_name2,
            hccl_comm_name3,
        )
        output = torch.ops.afd_ascend.e2a(
            ref_tensor,
            transfer_state.atten_batch_size,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            ffn_size,
            attn_size,
            world_rank,
            group_ep,
            transfer_state.aiv_num,
        )
        return output

    def recv_ffn_output_fake_impl(
        ref_tensor: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
    ) -> torch.Tensor:
        """Return the reference tensor while PyTorch inspects the receive."""
        return ref_tensor

    send_annotations = {
        "hidden_states": torch.Tensor,
        "hccl_comm_name": str,
        "hccl_comm_name2": str,
        "hccl_comm_name3": str,
        "batch_size": int,
        "hidden_size": int,
        "topk": int,
        "ffn_size": int,
        "attn_size": int,
        "world_rank": int,
        "aiv_num": int,
        "compute_gate": int,
        "expert_ids": torch.Tensor | None,
        "expert_scales": torch.Tensor | None,
        "return": torch.Tensor,
    }
    recv_annotations = {
        "ref_tensor": torch.Tensor,
        "hccl_comm_name": str,
        "hccl_comm_name2": str,
        "hccl_comm_name3": str,
        "batch_size": int,
        "hidden_size": int,
        "topk": int,
        "ffn_size": int,
        "attn_size": int,
        "world_rank": int,
        "aiv_num": int,
        "return": torch.Tensor,
    }
    send_attn_output_impl.__annotations__ = send_annotations
    send_attn_output_fake_impl.__annotations__ = send_annotations
    recv_ffn_output_impl.__annotations__ = recv_annotations
    recv_ffn_output_fake_impl.__annotations__ = recv_annotations

    try:
        direct_register_custom_op(
            op_name="afd_camp2p_send_attn_output",
            op_func=send_attn_output_impl,
            mutates_args=[],
            fake_impl=send_attn_output_fake_impl,
            dispatch_key="PrivateUse1",
        )
        direct_register_custom_op(
            op_name="afd_camp2p_recv_ffn_output",
            op_func=recv_ffn_output_impl,
            mutates_args=[],
            fake_impl=recv_ffn_output_fake_impl,
            dispatch_key="PrivateUse1",
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        duplicate = any(
            marker in message
            for marker in ("already", "duplicate", "same name", "defined")
        )
        if not duplicate:
            raise
    _CAMP2P_CUSTOM_OPS_REGISTERED = True


__all__ = [
    "CAMP2pAFDConnector",
    "CAMP2pAFDControlPlane",
    "CAMP2PAFDConnectorData",
    "CAMP2PExtraInfo",
    "CAMP2PTransferState",
    "build_camp2p_topology",
]

CAMP2PAFDConnectorData = CAMP2PTransferState
