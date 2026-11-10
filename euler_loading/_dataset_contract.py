"""Register the euler-dataset-contract validators euler-loading relies on.

The rules of the ``euler_loading`` head addon live in euler-dataset-contract
(``validate_euler_loading_addon``), so ds-crawler and other producers check
heads by the same rules euler-loading loads them with. The name stays
importable from here for compatibility.
"""

from __future__ import annotations

from euler_dataset_contract import (
    register_descriptor_validators,
    register_euler_loading_addon_validator,
    validate_euler_loading_addon,
)

__all__ = ["validate_euler_loading_addon"]

register_euler_loading_addon_validator()
register_descriptor_validators()
