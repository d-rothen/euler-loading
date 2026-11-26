"""Batches of :class:`~euler_loading.MultiModalDataset` samples.

A sample is a dict: one value per modality (a tensor or array, ``None`` for a
hierarchical modality without a file, a dict for an uncollapsed one), plus
``id``, ``full_id``, ``meta`` and ``attributes``. PyTorch's default collate
fails on that structure (``None``, per-modality metadata dicts whose keys
differ between samples), so every training loop wrote its own.
:func:`collate` is that function, once::

    from euler_loading import MultiModalDataset, dataloader

    loader = dataloader(dataset, batch_size=8, shuffle=True, num_workers=4)
    for batch in loader:
        batch["rgb"]        # (8, 3, H, W) tensor
        batch["full_id"]    # 8 ids, to match predictions back to their files

The rules, per key:

* tensors (or numpy arrays) of one shape are stacked along a new first axis,
  keeping their library and dtype; a mix of tensors and arrays becomes a
  tensor;
* ``id``, ``full_id``, ``meta`` and ``attributes`` stay one entry per sample:
  lists of strings and of the per-sample dicts;
* a modality missing from some samples (``None``, or the key absent) is a list
  with ``None`` at those positions, never stacked, so the gap stays visible;
* dicts (an uncollapsed hierarchical modality), strings, numbers and other
  values are lists;
* arrays of different shapes raise :class:`BatchError` naming the modality and
  shapes: resize them (``SamplePreprocessor``) or batch them one at a time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: Keys every sample carries besides its modalities; collated as lists.
SAMPLE_METADATA_KEYS = ("id", "full_id", "meta", "attributes")


class BatchError(ValueError):
    """Samples that cannot be batched together; the message says why and what to do."""


def _is_tensor(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("torch") and hasattr(value, "shape") and hasattr(value, "dtype")


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or _is_tensor(value)


def _stack(name: str, values: list[Any]) -> Any:
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) > 1:
        shown = ", ".join(str(list(shape)) for shape in sorted(shapes))
        raise BatchError(
            f"cannot batch {name!r}: its samples have different shapes ({shown}). "
            "Resize every modality to one size (euler_loading.SamplePreprocessor "
            "with a Resize, or data.resize in an euler-suite project), or use "
            "batch_size=1."
        )
    if any(_is_tensor(value) for value in values):
        import torch

        return torch.stack([torch.as_tensor(value) for value in values])
    return np.stack(values)


def collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One batch from a list of :class:`~euler_loading.MultiModalDataset` samples.

    See the module documentation for the rules. Keys appear in the order of
    the first sample, then any key only later samples carry.
    """
    if not samples:
        raise BatchError("cannot batch an empty list of samples")
    keys: list[str] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise BatchError(
                f"expected dataset samples (dicts), got {type(sample).__name__}"
            )
        keys.extend(key for key in sample if key not in keys)

    batch: dict[str, Any] = {}
    for key in keys:
        values = [sample.get(key) for sample in samples]
        if key in SAMPLE_METADATA_KEYS:
            batch[key] = values
        elif any(value is None for value in values):
            batch[key] = values
        elif all(_is_array(value) for value in values):
            batch[key] = _stack(key, values)
        else:
            batch[key] = values
    return batch


def dataloader(dataset: Any, *, batch_size: int = 1, **kwargs: Any) -> Any:
    """A ``torch.utils.data.DataLoader`` over *dataset* that batches with :func:`collate`.

    Every keyword argument except ``collate_fn`` goes to the ``DataLoader``
    (``shuffle``, ``num_workers``, ``pin_memory``, ...). Needs torch
    (``pip install euler-loading[gpu]``).
    """
    if "collate_fn" in kwargs:
        raise TypeError(
            "dataloader() batches with euler_loading.collate; build a DataLoader "
            "yourself for another collate_fn"
        )
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - torch is the gpu extra
        raise ImportError(
            "euler_loading.dataloader needs torch: pip install euler-loading[gpu]"
        ) from exc
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, **kwargs)


__all__ = ["BatchError", "SAMPLE_METADATA_KEYS", "collate", "dataloader"]
