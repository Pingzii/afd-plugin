# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for connector resolution through the factory.

The connector packages resolve their public names with a module-level
``__getattr__`` so importing one does not pull its runtime dependencies in. The
factory therefore has to read the class with ``getattr``; looking it up in the
module dictionary only works for eagerly exported names and failed for the
lazy CAMP2P package.
"""

from __future__ import annotations

import sys
import types

import pytest

from afd_plugin.connectors import factory
from afd_plugin.connectors.base import AFDConnectorBase


class _LazilyExportedConnector(AFDConnectorBase):
    """Stands in for a connector served only through module ``__getattr__``."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - never built
        raise NotImplementedError


@pytest.fixture()
def lazy_module(monkeypatch):
    """A module that answers ``getattr`` but never populates ``__dict__``."""

    module = types.ModuleType("afd_lazy_connector_module")
    class_name = "_LazilyExportedConnector"

    module.__dict__[class_name] = _LazilyExportedConnector
    # hide it from the dictionary again so only __getattr__ can serve it
    module.__dict__.pop(class_name)

    def _resolve(name: str):
        if name == class_name:
            return _LazilyExportedConnector
        raise AttributeError(name)

    # a module-level __getattr__ (PEP 562) is what the connector packages use
    module.__getattr__ = _resolve
    monkeypatch.setitem(sys.modules, "afd_lazy_connector_module", module)
    return module, class_name


def test_factory_resolves_a_class_that_only_lives_behind_getattr(lazy_module):
    module, class_name = lazy_module
    assert class_name not in vars(module), "the module dictionary must stay empty"

    factory.AFDConnectorFactory.register_connector(
        "LazilyResolvedConnector",
        module.__name__,
        class_name,
        replace=True,
    )

    resolved = factory.AFDConnectorFactory.get_connector_class(
        "LazilyResolvedConnector"
    )

    assert resolved is _LazilyExportedConnector


def test_factory_rejects_a_module_that_does_not_export_the_name(lazy_module):
    module, _ = lazy_module
    factory.AFDConnectorFactory.register_connector(
        "MissingNameConnector",
        module.__name__,
        "_NotExported",
        replace=True,
    )

    with pytest.raises(AttributeError):
        factory.AFDConnectorFactory.get_connector_class("MissingNameConnector")
