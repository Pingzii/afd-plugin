"""Activation gate for AFD configs.

Covers ``is_afd_active``, the predicate that decides whether an AFD config is
present *and* valid enough to activate, as opposed to merely present.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import AFD_ASYNC_CONNECTOR, is_afd_active, parse_afd_config


def _source(*, afd: dict | None) -> SimpleNamespace:
    """Build the minimal vLLM-config shape the config reader expects."""

    additional: dict = {}
    if afd is not None:
        additional["afd"] = afd
    return SimpleNamespace(
        additional_config=additional,
        parallel_config=SimpleNamespace(),
    )


def test_afd_config_without_the_key_is_inactive():
    assert is_afd_active(_source(afd=None)) is False


def test_valid_async_afd_config_is_active():
    source = _source(afd={"connector": AFD_ASYNC_CONNECTOR, "role": "attention", "async": True})

    assert is_afd_active(source) is True


def test_valid_ffn_role_is_active():
    source = _source(afd={"connector": AFD_ASYNC_CONNECTOR, "role": "ffn", "async": True})

    assert is_afd_active(source) is True


def test_unsupported_role_raises_instead_of_reporting_inactive():
    """A present-but-invalid role is a hard error, not simply "inactive"."""

    source = _source(afd={"connector": AFD_ASYNC_CONNECTOR, "role": "not-a-role", "async": True})

    with pytest.raises(ValueError, match="AFD role must be one of"):
        is_afd_active(source)


def test_missing_afd_key_raises_when_parsing_required_config():
    with pytest.raises(ValueError, match='additional_config\\["afd"\\]'):
        parse_afd_config(_source(afd=None))
