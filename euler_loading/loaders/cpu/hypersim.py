"""CPU-oriented loader functions for Hypersim, as prepared by euler-known-datasets.

Hypersim (Roberts et al., ICCV 2021, https://github.com/apple/ml-hypersim) is a
synthetic indoor dataset rendered with V-Ray. The ``euler-known-datasets``
recipe ``hypersim`` indexes its released files and adds one calibration file
per camera trajectory:

- ``rgb``: ``images/scene_cam_XX_final_hdf5/frame.IIII.color.hdf5``, linear HDR
  colour (float16), tone-mapped by :func:`rgb`;
- ``depth``: ``images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5``,
  which holds Euclidean *distances* to the camera centre in meters (not planar
  depth, despite the name); :func:`depth` converts them to planar depth;
- ``intrinsics`` and ``camera_pose``: ``calibration/<scene>/<camera>.json``.

Tone mapping
------------
:func:`rgb` applies Hypersim's own operator
(``code/python/tools/scene_generate_images_tonemap.py``): gamma ``1/2.2``
after scaling the image so that its 90th-percentile brightness
(``0.3 R + 0.59 G + 0.11 B``) becomes ``0.8``. The scale is computed by the
recipe over the pixels with geometry (``render_entity_id != -1``) and stored
in each file's ``tonemap_scale`` attribute; without it, all pixels count.

Camera model
------------
Many Hypersim scenes were rendered with tilt-shift lens settings, so the
stored camera axes are not the axes of the image (up to 41 degrees apart) and
the principal point is far from the image centre. The recipe therefore
derives, from the per-scene ``M_cam_from_uv`` matrix Hypersim publishes, an
exact pinhole camera: the matrix ``K`` (with skew where the lens is shifted
sideways) and its *optical frame* (x right, y down, z along the optical axis).
Poses map that frame into the scene's world frame, scaled to meters. Planar
depth is the optical-frame ``z``: ``distance / ||K^-1 [x, y, 1]||`` with pixel
centres at integer ``(x, y)``. Each depth file's ``intrinsics`` attribute
carries the ``K`` this conversion uses.

Calibration file
----------------
Format ``euler_known_datasets.hypersim.camera_frames``, version 1, documented
in euler-known-datasets' ``docs/datasets/hypersim.md``: ``camera.intrinsics``
(3x3) and ``frames[<frame id>].camera_to_world`` (4x4, row-major).

Return types
------------
- **rgb** -- ``(H, W, 3)`` float32 in ``[0, 1]``.
- **depth** -- ``(H, W)`` float32 planar depth in meters, ``0`` where no
  geometry was hit.
- **radial_distance** -- ``(H, W)`` float32 Euclidean distance in meters, ``0`` where
  no geometry was hit.
- **read_intrinsics** -- ``(3, 3)`` float32 (hierarchical).
- **camera_pose** -- ``(4, 4)`` float32 camera-to-world pose of the sample's
  frame, in meters (hierarchical, selected per sample).

Reading and writing HDF5 needs ``h5py`` (``pip install 'euler-loading[hypersim]'``).
Files written by this module carry an ``euler_encoding`` attribute, so a
written depth map or image is read back as written rather than converted
again. Readers accept paths and binary streams.
"""

from __future__ import annotations

import functools
import io
import json
import os
from collections.abc import Mapping
from typing import Any, BinaryIO, Union

import numpy as np

from euler_loading.loaders._annotations import (
    SampleContext,
    modality_meta,
    selects_per_sample,
)
from euler_loading.loaders._writer_utils import (
    ensure_parent,
    mark_stream_supported,
    pixel_grid_transform,
    replace_json,
    to_hw,
    to_hwc_rgb,
    to_numpy,
    write_json,
)

__all__ = [
    "FORMAT_NAME",
    "CameraTrajectory",
    "tonemap",
    "rgb",
    "depth",
    "radial_distance",
    "read_intrinsics",
    "camera_pose",
    "write_rgb",
    "write_depth",
    "write_radial_distance",
    "write_intrinsics",
]

#: The calibration file format these loaders read.
FORMAT_NAME = "euler_known_datasets.hypersim.camera_frames"
SUPPORTED_FORMAT_VERSIONS = (1,)

#: HDF5 attribute marking what a file written by this module holds.
ENCODING_ATTRIBUTE = "euler_encoding"
_SRGB = "srgb"
_PLANAR = "planar_depth_meters"
_DISTANCE = "distance_meters"

GAMMA = 1.0 / 2.2
PERCENTILE = 90
PERCENTILE_TARGET = 0.8
_CAMERA_AXES = "optical: x right, y down, z forward"


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Hypersim files are HDF5; install h5py (pip install 'euler-loading[hypersim]')"
        ) from exc
    return h5py


def _source_name(path: Union[str, BinaryIO]) -> str:
    return str(getattr(path, "name", path))


def _read_bytes(path: Union[str, BinaryIO]) -> bytes:
    if isinstance(path, (str, os.PathLike)):
        with open(path, "rb") as stream:
            return stream.read()
    return path.read()


def _read_hdf5(path: Union[str, BinaryIO]) -> tuple[np.ndarray, str | None]:
    """The array in a Hypersim HDF5 file's ``dataset`` and its ``euler_encoding``."""
    h5py = _h5py()
    source = path if isinstance(path, (str, os.PathLike)) else io.BytesIO(path.read())
    try:
        with h5py.File(source, "r") as handle:
            dataset = handle["dataset"]
            encoding = dataset.attrs.get(ENCODING_ATTRIBUTE)
            values = dataset[()]
    except (OSError, KeyError) as exc:
        raise ValueError(f"{_source_name(path)!r} is not a Hypersim HDF5 file: {exc}") from exc
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    return np.asarray(values), encoding


def _write_hdf5(path: Union[str, BinaryIO], values: np.ndarray, encoding: str) -> None:
    h5py = _h5py()
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as handle:
        dataset = handle.create_dataset("dataset", data=values, compression="gzip", compression_opts=4)
        dataset.attrs[ENCODING_ATTRIBUTE] = encoding
    data = buffer.getvalue()
    ensure_parent(path)
    if isinstance(path, (str, os.PathLike)):
        with open(path, "wb") as stream:
            stream.write(data)
        return
    path.write(data)


def _option(meta: Mapping[str, Any] | None, attributes: Mapping[str, Any] | None, key: str) -> Any:
    """Return *key* from the file's attributes, else the head's meta."""
    for source in (attributes, meta):
        if isinstance(source, Mapping) and source.get(key) is not None:
            return source[key]
    return None


def _matrix(value: Any, shape: tuple[int, int], what: str, source: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Hypersim {source!r}: {what} must be a finite {shape} matrix")
    return array


# ---------------------------------------------------------------------------
# Tone mapping and the camera model
# ---------------------------------------------------------------------------


def tonemap(color: np.ndarray, scale: float | None = None, valid: np.ndarray | None = None) -> np.ndarray:
    """Hypersim's tone-mapping operator: ``clip((scale * color) ** (1 / 2.2), 0, 1)``.

    Without *scale*, it is chosen so that the 90th-percentile brightness of the
    *valid* pixels (all finite ones when ``None``) becomes ``0.8``, as
    ``scene_generate_images_tonemap.py`` does with ``render_entity_id != -1``.
    """
    color = np.where(np.isfinite(color), color, 0).astype(np.float32, copy=False)
    if scale is None:
        brightness = 0.3 * color[..., 0] + 0.59 * color[..., 1] + 0.11 * color[..., 2]
        if valid is not None:
            brightness = brightness[np.asarray(valid, dtype=bool)]
        if brightness.size == 0:
            scale = 1.0
        else:
            current = float(np.percentile(brightness, PERCENTILE))
            scale = 0.0 if current < 1e-4 else PERCENTILE_TARGET ** (1.0 / GAMMA) / current
    return np.clip(np.power(np.maximum(np.float32(scale) * color, 0), np.float32(GAMMA)), 0, 1)


def _intrinsics_attribute(attributes: Mapping[str, Any] | None, source: str) -> np.ndarray:
    value = attributes.get("intrinsics") if isinstance(attributes, Mapping) else None
    if value is None:
        raise ValueError(
            f"{source!r} holds Hypersim distances; converting them needs the camera matrix "
            "in the file's 'intrinsics' attribute, which the euler-known-datasets recipe "
            "'hypersim' writes (or load the 'radial_distance' function instead)"
        )
    return _matrix(value, (3, 3), "attributes.intrinsics", source)


@functools.lru_cache(maxsize=64)
def _ray_lengths(matrix: tuple[float, ...], height: int, width: int) -> np.ndarray:
    """``||K^-1 [x, y, 1]||`` for every pixel centre: distance / planar depth."""
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    rays = inverse[:, 0, None, None] * xs + inverse[:, 1, None, None] * ys + inverse[:, 2, None, None]
    rays /= rays[2]
    lengths = np.sqrt((rays**2).sum(axis=0)).astype(np.float32)
    lengths.setflags(write=False)
    return lengths


def _ray_lengths_for(matrix: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return _ray_lengths(tuple(float(v) for v in matrix.ravel()), int(shape[0]), int(shape[1]))


def _clean_depth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(values) & (values > 0), values, np.float32(0.0))


# ---------------------------------------------------------------------------
# Calibration file
# ---------------------------------------------------------------------------


def _read_document(path: Union[str, BinaryIO]) -> dict[str, Any]:
    source = _source_name(path)
    try:
        document = json.loads(_read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Hypersim calibration {source!r} is not JSON: {exc}") from exc
    file_format = document.get("format") if isinstance(document, dict) else None
    if not isinstance(file_format, Mapping) or file_format.get("name") != FORMAT_NAME:
        raise ValueError(
            f"{source!r} is not a Hypersim calibration file ({FORMAT_NAME}); prepare the "
            "dataset with euler-known-datasets"
        )
    if file_format.get("version") not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Hypersim calibration {source!r} has format version {file_format.get('version')!r}; "
            f"this euler-loading reads {', '.join(map(str, SUPPORTED_FORMAT_VERSIONS))}"
        )
    return document


class CameraTrajectory:
    """A parsed calibration file: one camera trajectory of one scene.

    Attributes:
        scene: The scene name, e.g. ``"ai_001_001"``.
        camera: The trajectory name, e.g. ``"cam_00"``.
        intrinsics: ``(3, 3)`` float64 camera matrix.
        frames: The file's frame records, keyed by frame ID (``"0000"``).
    """

    def __init__(self, document: Mapping[str, Any], *, source: str = "<calibration>"):
        self.source = source
        self.scene = str(document.get("scene", ""))
        camera = document.get("camera") or {}
        self.camera = str(camera.get("name", ""))
        self.intrinsics = _matrix(camera.get("intrinsics"), (3, 3), "camera.intrinsics", source)
        self.frames: Mapping[str, Mapping[str, Any]] = document.get("frames") or {}

    def camera_to_world(self, frame_id: str) -> np.ndarray:
        """The ``(4, 4)`` float64 pose of the frame's optical frame, in meters."""
        record = self.frames.get(frame_id)
        if record is None:
            raise KeyError(
                f"Hypersim calibration {self.source!r} ({self.scene} {self.camera}) has no "
                f"frame {frame_id!r}; the sample ID must be the frame number"
            )
        return _matrix(record["camera_to_world"], (4, 4), f"frames[{frame_id}].camera_to_world", self.source)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


@modality_meta(
    modality_type="rgb",
    dtype="float32",
    shape="HWC",
    file_formats=[".hdf5"],
    output_range=[0.0, 1.0],
    meta={"stored": "linear HDR colour", "tonemap": "hypersim"},
)
def rgb(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``color.hdf5`` image tone-mapped as ``(H, W, 3)`` float32 in ``[0, 1]``.

    The scale comes from the file's ``tonemap_scale`` attribute (see the module
    docstring). An image written by :func:`write_rgb` is returned as written.
    """
    values, encoding = _read_hdf5(path)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"Hypersim colour {_source_name(path)!r} has shape {values.shape}, not (H, W, 3)")
    if encoding == _SRGB:
        return np.clip(values.astype(np.float32), 0, 1)
    scale = _option(None, attributes, "tonemap_scale")
    return tonemap(values.astype(np.float32), None if scale is None else float(scale))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".hdf5"],
    output_unit="meters",
    meta={"radial_depth": False, "scale_to_meters": 1.0, "stored": "euclidean distance"},
)
def depth(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``depth_meters.hdf5`` map as ``(H, W)`` float32 planar depth in meters.

    The released distances are divided by ``||K^-1 [x, y, 1]||`` with ``K``
    from the file's ``intrinsics`` attribute. Pixels without geometry are
    ``0``. A map written by :func:`write_depth` is returned as written.
    """
    values, encoding = _read_hdf5(path)
    values = values.astype(np.float32)
    if encoding == _PLANAR:
        return _clean_depth(values)
    matrix = _intrinsics_attribute(attributes, _source_name(path))
    return _clean_depth(values / _ray_lengths_for(matrix, values.shape))


@modality_meta(
    modality_type="depth",
    dtype="float32",
    shape="HW",
    file_formats=[".hdf5"],
    output_unit="meters",
    meta={"radial_depth": True, "scale_to_meters": 1.0, "stored": "euclidean distance"},
)
def radial_distance(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Load a ``depth_meters.hdf5`` map as ``(H, W)`` float32 Euclidean distance
    to the camera centre in meters, as released. Pixels without geometry are ``0``.
    """
    values, encoding = _read_hdf5(path)
    values = values.astype(np.float32)
    if encoding == _PLANAR:
        matrix = _intrinsics_attribute(attributes, _source_name(path))
        values = values * _ray_lengths_for(matrix, values.shape)
    return _clean_depth(values)


@modality_meta(
    modality_type="intrinsics",
    dtype="float32",
    hierarchical=True,
    shape="3x3",
    file_formats=[".json"],
    meta={"model": "pinhole", "distortion": None, "source": "calibration/<scene>/<camera>.json"},
)
def read_intrinsics(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> np.ndarray:
    """Return the trajectory's ``(3, 3)`` float32 camera matrix.

    It is exact for every scene, including the tilt-shift ones, whose
    principal point lies far from the image centre and whose matrix may have
    skew (``K[0, 1]``, up to 16 px). Index the file at the scene/camera level.
    """
    return CameraTrajectory(_read_document(path), source=_source_name(path)).intrinsics.astype(np.float32)


def _select_pose(table: CameraTrajectory, context: SampleContext) -> np.ndarray:
    return table.camera_to_world(context.id).astype(np.float32)


@selects_per_sample(_select_pose)
@modality_meta(
    modality_type="camera_extrinsics",
    dtype="float32",
    hierarchical=True,
    shape="4x4",
    file_formats=[".json"],
    meta={
        "transform": "camera_to_world",
        "source_frame": "camera_optical",
        "target_frame": "world",
        "camera_axes": _CAMERA_AXES,
        "translation_unit": "meters",
        "matched_by": "sample ID (the frame number)",
    },
)
def camera_pose(path: Union[str, BinaryIO], meta: dict[str, Any] | None = None, *, attributes: dict[str, Any] | None = None) -> CameraTrajectory:
    """Load a calibration file; each sample receives its ``(4, 4)`` float32
    camera-to-world pose in meters.

    The pose maps the optical frame (x right, y down, z forward; see the
    module docstring) into the scene's world frame, whose asset units are
    scaled to meters. Calling the loader directly returns the whole
    :class:`CameraTrajectory`.
    """
    return CameraTrajectory(_read_document(path), source=_source_name(path))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


@mark_stream_supported
def write_rgb(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write an ``(H, W, 3)`` or ``(3, H, W)`` image in ``[0, 1]`` as HDF5 (float32).

    :func:`rgb` returns it unchanged: it is marked as already tone-mapped.
    """
    values = np.clip(np.asarray(to_hwc_rgb(value, name="rgb"), dtype=np.float32), 0, 1)
    _write_hdf5(path, values, _SRGB)


@mark_stream_supported
def write_depth(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write planar depth in meters as HDF5 (float32), as :func:`depth` returns it.

    Non-finite and non-positive values become ``0``.
    """
    _write_hdf5(path, _clean_depth(to_hw(value, name="depth")), _PLANAR)


@mark_stream_supported
def write_radial_distance(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write Euclidean distances in meters as HDF5 (float32), as :func:`radial_distance` returns them."""
    _write_hdf5(path, _clean_depth(to_hw(value, name="radial_distance")), _DISTANCE)


def _image_size(meta: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(meta, Mapping) or meta.get("width") is None or meta.get("height") is None:
        return None
    return int(meta["width"]), int(meta["height"])


def write_intrinsics(path: Union[str, BinaryIO], value: Any, meta: dict[str, Any] | None = None) -> None:
    """Write a camera matrix into a Hypersim calibration file.

    *value* is what :func:`read_intrinsics` returns. An existing file is
    updated, not replaced: the change from its stored matrix must be a crop or
    a rescale, ``meta["width"]`` and ``meta["height"]`` update the recorded
    image size, and the poses stay as they are. An unchanged matrix leaves the
    file byte-identical. A new file, or a stream target, receives a calibration
    file with only the camera matrix (and size).
    """
    matrix = to_numpy(value).astype(np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"intrinsics must be a finite (3, 3) matrix, got shape {matrix.shape}")
    size = _image_size(meta)

    if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
        name = os.fspath(path)
        document = _read_document(path)
        stored = CameraTrajectory(document, source=name).intrinsics
        transform = pixel_grid_transform(matrix @ np.linalg.inv(stored))
        if transform is None:
            raise ValueError(
                f"Refusing to update {name!r}: the new intrinsics are not a crop or rescale "
                f"of the stored matrix ({stored.tolist()} -> {matrix.tolist()})."
            )
        camera = document["camera"]
        updated = dict(camera, intrinsics=(transform @ stored).tolist())
        if size is not None:
            updated["width"], updated["height"] = size
        if updated == camera:
            return
        document["camera"] = updated
        replace_json(path, document)
        return

    camera: dict[str, Any] = {"intrinsics": matrix.tolist()}
    if size is not None:
        camera["width"], camera["height"] = size
    write_json(path, {"format": {"name": FORMAT_NAME, "version": 1}, "camera": camera, "frames": {}})
