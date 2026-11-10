"""CPU-oriented loader functions for the ScatterNeRF dataset.

ScatterNeRF (Ramazzina et al., ICCV 2023) releases foggy driving sequences in
the NSVF layout used by NeRF pipelines::

    <sequence>/rgb/<camera>_<frame>_<timestamp ns>.png        # left / right
    <sequence>/intrinsics/<camera>_<frame>_<timestamp ns>.txt  # 4x4
    <sequence>/pose/<camera>_<frame>_<timestamp ns>.txt        # 4x4 camera-to-world

The images are the rectified SeeingThroughFog stereo pair (``fx = fy =
2355.7228``), cropped to 1704x908 with the principal point at the image centre.
Intrinsics are therefore an ideal pinhole without distortion.

Poses map points from the camera, in optical axes (x right, y down,
z forward), into a world frame chosen per sequence. Like most NeRF data, each
sequence is scaled to its own normalized units: its stereo baseline measures
0.021 to 0.038 units for a 0.203 m rig. euler-known-datasets records each
sequence's metric scale in the files' ``scale_to_meters`` attribute, which
:func:`camera_pose_metric` applies.

The release has no lidar.

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.
- **read_intrinsics** -- ``(3, 3)`` float32 camera matrix (per image).
- **camera_pose** -- ``(4, 4)`` float64 camera-to-world transform (per image),
  in stored units; **camera_pose_metric** in meters.

All readers accept paths and binary streams.
"""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image

from euler_loading.loaders._annotations import modality_meta
from euler_loading.loaders._writer_utils import (
    get_target_name,
    mark_stream_supported,
    pixel_grid_transform,
    save_image,
    to_hwc_rgb,
    to_numpy,
    to_uint8,
    write_text,
)

__all__ = [
    "rgb",
    "read_intrinsics",
    "camera_pose",
    "camera_pose_metric",
    "write_rgb",
    "write_intrinsics",
    "write_camera_pose",
]

_RIGID_TOLERANCE = 1e-4


def _source_name(path: Union[str, BinaryIO]) -> str:
    return str(getattr(path, "name", path))


def _read_bytes(path: Union[str, BinaryIO]) -> bytes:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return path.read()


def _option(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None, key: str) -> Any:
    """Return *key* from the file's attributes, else the head's meta."""
    for source in (attributes, meta):
        if isinstance(source, Mapping) and source.get(key) is not None:
            return source[key]
    return None


def _read_matrix(path: Union[str, BinaryIO], what: str) -> np.ndarray:
    source = _source_name(path)
    try:
        values = np.array(_read_bytes(path).decode("utf-8").split(), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"ScatterNeRF {what} {source!r} is not a list of numbers") from exc
    if values.size != 16 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"ScatterNeRF {what} {source!r} must hold 16 finite numbers (a 4x4 matrix), "
            f"found {values.size}"
        )
    return values.reshape(4, 4)


def _format(matrix: np.ndarray) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(matrix, dtype=np.float64).ravel()) + "\n"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".png"],
    output_range=[0.0, 1.0],
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a rectified stereo frame (1704x908 PNG) as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    with Image.open(io.BytesIO(_read_bytes(path))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    shape="3x3",
    file_formats=[".txt"],
    meta={"model": "pinhole", "distortion": None, "stored_as": "4x4 NSVF intrinsics, row-major"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the ``(3, 3)`` float32 camera matrix of one image.

    The file stores the NSVF 4x4 ``[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0],
    [0, 0, 0, 1]]``; its upper-left 3x3 block is the pinhole matrix.
    """
    matrix = _read_matrix(path, "intrinsics")
    K = matrix[:3, :3]
    if not (np.allclose(matrix[3], (0, 0, 0, 1)) and np.allclose(matrix[:3, 3], 0)
            and np.allclose(K[2], (0, 0, 1)) and K[0, 1] == 0 and K[1, 0] == 0
            and K[0, 0] > 0 and K[1, 1] > 0):
        raise ValueError(
            f"ScatterNeRF intrinsics {_source_name(path)!r} are not a pinhole matrix "
            f"padded to 4x4: {matrix.tolist()}"
        )
    return K.astype(np.float32)


_POSE_META = {
    "transform": "camera_to_world",
    "source_frame": "camera",
    "target_frame": "world",
    "camera_axes": "optical: x right, y down, z forward",
    "world": "per sequence",
}


def _read_pose(path: Union[str, BinaryIO]) -> np.ndarray:
    pose = _read_matrix(path, "pose")
    rotation = pose[:3, :3]
    if not (np.allclose(pose[3], (0, 0, 0, 1))
            and np.allclose(rotation @ rotation.T, np.eye(3), atol=_RIGID_TOLERANCE)
            and np.linalg.det(rotation) > 0):
        raise ValueError(f"ScatterNeRF pose {_source_name(path)!r} is not a rigid transform")
    return pose


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float64",
    shape="4x4",
    file_formats=[".txt"],
    meta={**_POSE_META, "translation_unit": "per-sequence normalized units, as stored"},
)
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the ``(4, 4)`` float64 camera-to-world pose of one image, as stored.

    Translations are in the sequence's normalized units; see
    :func:`camera_pose_metric` for meters. Relative motion between two frames
    is ``inv(pose_a) @ pose_b``; for a stereo pair it is the baseline.
    """
    return _read_pose(path)


@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float64",
    shape="4x4",
    file_formats=[".txt"],
    meta={**_POSE_META, "translation_unit": "meters", "requires": "scale_to_meters"},
)
def camera_pose_metric(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the ``(4, 4)`` float64 camera-to-world pose with translations in meters.

    The stored translation is multiplied by ``scale_to_meters``, read from the
    file's attributes and then the head's meta. euler-known-datasets records
    it per sequence from the 0.203 m stereo baseline. Select it with
    ``Modality(root, metadata_scope="camera_pose", function="camera_pose_metric")``.
    It has no writer: the per-sequence scale lives in the index, not the file,
    so write :func:`camera_pose` values instead.
    """
    scale = _option(meta, attributes, "scale_to_meters")
    if scale is None or not float(scale) > 0:
        raise ValueError(
            f"camera_pose_metric needs a positive scale_to_meters for {_source_name(path)!r}; "
            "euler-known-datasets records it per sequence in the file attributes"
        )
    pose = _read_pose(path)
    pose[:3, 3] *= float(scale)
    return pose


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an RGB image as 8-bit PNG."""
    image = Image.fromarray(to_uint8(to_hwc_rgb(value, name="rgb"), scale_unit_range=True), mode="RGB")
    if os.path.splitext(get_target_name(path))[1].lower() not in (".png", ""):
        raise ValueError(f"ScatterNeRF images are PNG, not {get_target_name(path)!r}")
    save_image(path, image, format="PNG")


@mark_stream_supported
def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(3, 3)`` camera matrix as the NSVF 4x4 intrinsics file.

    Every image has its own file, so a crop or resize rewrites it whole. The
    matrix must still be a pinhole: zero skew and a last row of ``(0, 0, 1)``.
    """
    K = to_numpy(value).astype(np.float64)
    if K.shape != (3, 3) or pixel_grid_transform(K) is None:
        raise ValueError(f"ScatterNeRF intrinsics must be a (3, 3) pinhole matrix, got {K.tolist()}")
    matrix = np.eye(4)
    matrix[:3, :3] = K
    write_text(path, _format(matrix))


@mark_stream_supported
def write_camera_pose(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a ``(4, 4)`` camera-to-world pose as the NSVF pose file.

    The matrix is written as given. A crop or resize leaves poses unchanged;
    write stored-unit poses unless the written dataset declares meters.
    """
    pose = to_numpy(value).astype(np.float64)
    if pose.shape != (4, 4) or not np.allclose(pose[3], (0, 0, 0, 1)):
        raise ValueError(f"ScatterNeRF poses must be (4, 4) rigid transforms, got shape {pose.shape}")
    write_text(path, _format(pose))
