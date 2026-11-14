# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Helpers for tests that import the plugin under stand-in modules."""

from __future__ import annotations

import sys


def drop_stub_bound_modules(before: set[str], stub_objects: set[int]) -> None:
    """Forget modules the stub block loaded that still hold one of its objects.

    A cached module keeps the doubles in its globals, and the parent package keeps
    the child module as an attribute, so both have to be dropped for a later
    import to build the real thing.
    """

    for name in [
        name
        for name in sys.modules
        if name not in before and name.startswith("afd_plugin")
    ]:
        module = sys.modules[name]
        if not any(
            value is not None and id(value) in stub_objects
            for value in vars(module).values()
        ):
            continue
        sys.modules.pop(name, None)
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, leaf, None) is module:
            delattr(parent, leaf)
