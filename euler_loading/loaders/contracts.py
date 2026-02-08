"""Loader contracts (Protocols) for common dataset types.

Each protocol defines the set of loader functions a dataset module must
provide.  Protocols are :func:`~typing.runtime_checkable`, so you can
verify conformance at runtime::

    from euler_loading.loaders.gpu import vkitti2
    from euler_loading.loaders.contracts import DenseDepthLoader

    assert isinstance(vkitti2, DenseDepthLoader)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class DenseDepthLoader(Protocol):
    """Contract for dense-depth dataset loaders.

    A conforming module must expose the following callables, each accepting
    a file path and returning a ``torch.Tensor``:

    - **rgb** – ``(3, H, W)`` float32 in ``[0, 1]``
    - **depth** – ``(1, H, W)`` float32 in metres
    - **sky_mask** – ``(1, H, W)`` bool
    - **read_intrinsics** – ``(3, 3)`` float32 camera matrix *K*
    """

    def rgb(self, path: str) -> torch.Tensor: ...
    def depth(self, path: str) -> torch.Tensor: ...
    def sky_mask(self, path: str) -> torch.Tensor: ...
    def read_intrinsics(self, path: str) -> torch.Tensor: ...
