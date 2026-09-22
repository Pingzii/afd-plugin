# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validation for token ids that index a token-keyed (Hash) routing table.

The CANN Hash operator indexes its token-to-expert table with the raw id of each
row it routes and never validates that id, so an out-of-range value faults the
AIV core with an MTE DDR error that names only the core. This module turns that
into an error that names the offending rows, which is what separates an id buffer
that was never fully written from a table that is smaller than the id space.

It deliberately depends on ``torch`` and ``afd_plugin.envs`` only: both the model
routing path and the NPU connectors validate ids, and neither should have to
import the other.
"""

from __future__ import annotations

import torch

from afd_plugin.envs import validate_hash_token_ids_enabled


def validate_hash_token_ids(
    input_ids: torch.Tensor,
    *,
    table_rows: int,
    context: str,
    padding_value: int | None = None,
) -> None:
    """Fail fast when Hash routing ids cannot index the ``tid2eid`` table.

    Disabled unless ``AFD_VALIDATE_HASH_TOKEN_IDS=1``, because the check reads
    device tensors and therefore synchronises.

    Args:
        input_ids: Ids about to be handed to the Hash operator.
        table_rows: Row count of the ``tid2eid`` table they index.
        context: Human-readable description of the routing path, used in the
            error message.
        padding_value: Optional sentinel that the routing path maps to token 0
            before the lookup, so rows carrying it never reach the table.

    Raises:
        RuntimeError: If the table has no rows, or if any id is outside it.
    """

    if not validate_hash_token_ids_enabled():
        return
    rows = int(table_rows)
    if rows <= 0:
        raise RuntimeError(
            f"{context}: the Hash token-to-expert table has {rows} rows, so no "
            "id can be routed through it",
        )
    ids = input_ids.reshape(-1)
    invalid = (ids < 0) | (ids >= rows)
    if padding_value is not None:
        invalid = invalid & (ids != int(padding_value))
    invalid_count = int(invalid.sum())
    if invalid_count == 0:
        return
    invalid_index = invalid.nonzero().flatten()[:8]
    raise RuntimeError(
        f"{context}: {invalid_count} of {int(ids.numel())} Hash routing ids are "
        f"outside [0, {rows}) and would fault the tid2eid lookup on device; "
        f"rows={invalid_index.tolist()} values={ids[invalid_index].tolist()} "
        f"min={int(ids.min())} max={int(ids.max())} dtype={ids.dtype}",
    )


__all__ = ["validate_hash_token_ids"]
