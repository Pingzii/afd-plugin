# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Runtime gate for the sync Attention-side gate prototype.

The sync shell does not call the DBO yield, so a DBO config would interleave
microbatches against a transport that never yields. Not calling the yield is not
the same as forbidding the feature, so the limit is enforced here.
"""

from types import SimpleNamespace

import pytest

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
