# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 A2E MXFP8 contract checks that do not need an NPU or vLLM-Ascend.

The transport itself is exercised on hardware; these cover the pure decisions
AFD makes around it, which is where a wrong answer is silent rather than loud.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.feature_validation import _fail_if_unsupported_a2e_mxfp8
from afd_plugin.compat.patches.npu.moe_mxfp8_scale import (
    AFD_MXFP8_SCALE_ATTR,
    apply_afd_moe_mxfp8_scale_patch,
)

_A2E_MX_MAX_TOTAL_RANKS = 64


def _afd_config(*, attention_ranks: int, ffn_ranks: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_attention_ranks=attention_ranks,
        num_ffn_ranks=ffn_ranks,
    )


class TestA2eMxfp8Topology:
    """The operator carries the A5 restriction; the rank shape is ours to check."""

    def test_accepts_equal_rank_counts(self):
        _fail_if_unsupported_a2e_mxfp8(
            _afd_config(attention_ranks=4, ffn_ranks=4),
            _A2E_MX_MAX_TOTAL_RANKS,
        )

    def test_accepts_the_total_rank_boundary(self):
        _fail_if_unsupported_a2e_mxfp8(
            _afd_config(attention_ranks=32, ffn_ranks=32),
            _A2E_MX_MAX_TOTAL_RANKS,
        )

    @pytest.mark.parametrize(
        ("attention_ranks", "ffn_ranks"),
        [(4, 2), (2, 4), (1, 8), (8, 1)],
    )
    def test_rejects_unequal_rank_counts(self, attention_ranks, ffn_ranks):
        with pytest.raises(RuntimeError, match="equal Attention and FFN rank counts"):
            _fail_if_unsupported_a2e_mxfp8(
                _afd_config(
                    attention_ranks=attention_ranks,
                    ffn_ranks=ffn_ranks,
                ),
                _A2E_MX_MAX_TOTAL_RANKS,
            )

    def test_rejects_more_than_the_total_rank_bound(self):
        with pytest.raises(RuntimeError, match="at most 64 ranks"):
            _fail_if_unsupported_a2e_mxfp8(
                _afd_config(attention_ranks=64, ffn_ranks=64),
                _A2E_MX_MAX_TOTAL_RANKS,
            )

    def test_unequal_counts_are_reported_before_the_total_bound(self):
        """A 4A8F config is both unequal and over nothing; name the real mistake."""

        with pytest.raises(RuntimeError, match="equal Attention and FFN rank counts"):
            _fail_if_unsupported_a2e_mxfp8(
                _afd_config(attention_ranks=4, ffn_ranks=8),
                _A2E_MX_MAX_TOTAL_RANKS,
            )


class TestA2eMxfp8ScaleResolver:
    """The resolver is installed by the plugin and must degrade, never raise."""

    def test_the_attribute_name_is_afd_owned(self):
        assert AFD_MXFP8_SCALE_ATTR == "afd_mxfp8_scale"

    def test_install_is_idempotent_or_reports_unavailable(self):
        # Either vLLM-Ascend is importable and the resolver installs, or it is
        # not and the call reports failure. It must not raise: only quant_mode=4
        # needs it, and the connector turns a missing resolver into a startup
        # error at that point.
        first = apply_afd_moe_mxfp8_scale_patch()
        second = apply_afd_moe_mxfp8_scale_patch()

        assert isinstance(first, bool)
        assert second is first or (first and second)
