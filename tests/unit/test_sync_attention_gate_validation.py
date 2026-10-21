# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Runtime gate for the sync Attention-side gate prototype.

The sync shell does not call the DBO yield, so a DBO config would interleave
microbatches against a transport that never yields. Not calling the yield is not
the same as forbidding the feature, so the limit is enforced here.

The entry-level tests matter more than the helper ones: a helper test keeps passing
if the call is deleted from the validator, while a test that goes through the
validator fails. Both roles are covered because both run this validator.
"""

from types import SimpleNamespace

import pytest

from afd_plugin import validation
from afd_plugin.validation import _fail_if_unsupported_sync_gate_runtime


def _config(*, enable_dbo=False, use_ubatching=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_dbo=enable_dbo,
            use_ubatching=use_ubatching,
        ),
    )


def test_plain_sync_gate_config_is_accepted():
    _fail_if_unsupported_sync_gate_runtime(_config(), SimpleNamespace())


def test_dbo_is_rejected():
    with pytest.raises(RuntimeError, match="DBO or ubatching"):
        _fail_if_unsupported_sync_gate_runtime(_config(enable_dbo=True), SimpleNamespace())


def test_ubatching_is_rejected():
    with pytest.raises(RuntimeError, match="DBO or ubatching"):
        _fail_if_unsupported_sync_gate_runtime(
            _config(use_ubatching=True),
            SimpleNamespace(),
        )


def _runner_config(*, enable_dbo=False):
    """Minimal ``VllmConfig`` surface reached before the DBO check."""

    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(
            enable_dbo=enable_dbo,
            use_ubatching=False,
        ),
    )


@pytest.fixture()
def sync_gate_afd_config(monkeypatch):
    """Make the validator see the sync gate-on-Attention configuration."""

    from afd_plugin.config import CAMP2P_CONNECTOR

    afd_config = SimpleNamespace(
        connector=CAMP2P_CONNECTOR,
        async_dp=False,
        compute_gate_on_attention=True,
    )
    monkeypatch.setattr(
        validation,
        "parse_afd_config",
        lambda _vllm_config, expected_role=None: afd_config,
    )
    return afd_config


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_validator_rejects_dbo_for_each_role(sync_gate_afd_config, role):
    """Both roles run this validator, so both must refuse DBO."""

    with pytest.raises(RuntimeError, match="DBO or ubatching"):
        validation.validate_npu_model_runner_v2_config(
            _runner_config(enable_dbo=True),
            expected_role=role,
            device_type="npu",
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_validator_gets_past_the_gate_for_a_plain_config(
    sync_gate_afd_config,
    role,
):
    """With DBO off the validator must not report the gate as unsupported.

    The fake config stops supplying fields once the validator walks past the gate
    branches, so a later AttributeError still counts as "the gate let it through".
    What matters here is that neither gate message appears.
    """

    try:
        validation.validate_npu_model_runner_v2_config(
            _runner_config(),
            expected_role=role,
            device_type="npu",
        )
    except RuntimeError as exc:
        assert "DBO or ubatching" not in str(exc)
        assert "requires synchronous CAMP2pAFDConnector" not in str(exc)
    except AttributeError:
        # Reached a field this fake does not model, which is past the gate.
        return
