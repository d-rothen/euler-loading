"""Shared pinhole image geometry; pixel centers are zero-based half pixels."""

from __future__ import annotations

import numpy as np


def resize_matrix(source_size, target_size):
    sy, sx = target_size[0] / source_size[0], target_size[1] / source_size[1]
    return ((sx, 0.0, (sx - 1) / 2), (0.0, sy, (sy - 1) / 2), (0.0, 0.0, 1.0))


def crop_matrix(top, left):
    return ((1.0, 0.0, float(-left)), (0.0, 1.0, float(-top)), (0.0, 0.0, 1.0))


def transform_intrinsics(intrinsics, matrix):
    if intrinsics.ndim not in (2, 3) or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(
            f"Expected intrinsics with shape (3, 3) or (N, 3, 3), got {tuple(intrinsics.shape)}"
        )
    if hasattr(intrinsics, "detach"):
        import torch

        dtype = intrinsics.dtype if intrinsics.is_floating_point() else torch.float64
        matrix = torch.as_tensor(matrix, dtype=dtype, device=intrinsics.device)
        return (matrix @ intrinsics.to(dtype)).to(intrinsics.dtype)
    dtype = (
        intrinsics.dtype if np.issubdtype(intrinsics.dtype, np.floating) else np.float64
    )
    matrix = np.asarray(matrix, dtype=dtype)
    return (matrix @ np.asarray(intrinsics, dtype=dtype)).astype(
        intrinsics.dtype, copy=False
    )
