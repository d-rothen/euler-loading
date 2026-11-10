from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image

try:  # pragma: no cover - exercised in torch-enabled environments
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised in CPU-only environments
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


__all__ = [
    "FieldSpec",
    "MaskedValueOverride",
    "Resize",
    "Crop",
    "SamplePreprocessor",
    "infer_field_spec",
    "resize_intrinsics",
    "resize_raster",
    "crop_intrinsics",
]


_PIL_RESAMPLING = getattr(Image, "Resampling", Image)
_PIL_MODES = {
    "nearest": _PIL_RESAMPLING.NEAREST,
    "bilinear": _PIL_RESAMPLING.BILINEAR,
    "bicubic": _PIL_RESAMPLING.BICUBIC,
}
# "sparse" forward-maps valid samples instead of interpolating; see _resize_sparse.
_INTERPOLATIONS = (*_PIL_MODES, "sparse")
# torch's "nearest" is floor-based; "nearest-exact" samples pixel centres like
# Pillow's NEAREST and the align_corners=False convention of resize_intrinsics.
_TORCH_MODES = {"nearest": "nearest-exact", "bilinear": "bilinear", "bicubic": "bicubic"}
_FIELD_KIND_ALIASES = {
    "auto": "generic",
    "camera_rays": "ray_map",
    "disparity": "disparity",
    "image": "image",
    "intrinsic": "intrinsics",
    "intrinsics": "intrinsics",
    "mask": "mask",
    "passthrough": "passthrough",
    "ray": "ray_map",
    "ray_map": "ray_map",
    "rays": "ray_map",
    "rgb": "image",
    "segmentation": "mask",
    "sky_mask": "mask",
    "spherical_map": "ray_map",
    "stereo_disparity": "disparity",
}
_DEFAULT_INTERPOLATION = {
    "depth": "bilinear",
    "disparity": "bilinear",
    "generic": None,
    "image": "bilinear",
    "intrinsics": None,
    "mask": "nearest",
    "passthrough": None,
    "ray_map": "bilinear",
}
_SUPPORTED_LAYOUTS = {"HW", "CHW", "HWC", "NHW", "NCHW", "NHWC"}


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _is_array_like(value: Any) -> bool:
    return isinstance(value, np.ndarray) or _is_torch_tensor(value)


def _canonical_kind(kind: str | None) -> str:
    if kind is None:
        return "generic"
    normalized = str(kind).strip().lower()
    return _FIELD_KIND_ALIASES.get(normalized, normalized)


def _parse_size(value: Any, *, context: str) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, Mapping):
        size = value.get("size", value.get("target_size"))
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])
    raise ValueError(f"{context} must be [height, width] or {{size: [height, width]}}.")


def _canonical_layout(layout: str | None) -> str | None:
    if layout is None:
        return None
    normalized = str(layout).strip().upper()
    if normalized not in _SUPPORTED_LAYOUTS:
        available = ", ".join(sorted(_SUPPORTED_LAYOUTS))
        raise ValueError(f"Unsupported layout {layout!r}. Available layouts: {available}")
    return normalized


def _infer_kind_from_name(name: str) -> str | None:
    lowered = name.lower()
    if "intrinsic" in lowered:
        return "intrinsics"
    if "ray" in lowered:
        return "ray_map"
    if "mask" in lowered or "segmentation" in lowered:
        return "mask"
    if "rgb" in lowered or "image" in lowered:
        return "image"
    if "disparity" in lowered:
        return "disparity"
    if "depth" in lowered:
        return "depth"
    return None


def _check_interpolation(interpolation: Any) -> str:
    mode = str(interpolation).strip().lower()
    if mode not in _INTERPOLATIONS:
        available = ", ".join(sorted(_INTERPOLATIONS))
        raise ValueError(
            f"Unsupported interpolation {interpolation!r}. "
            f"Available interpolations: {available}"
        )
    return mode


def _check_invalid_value(value: Any) -> float | None:
    """Return *value* as a float: a finite sentinel, or NaN for NaN-marked holes."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "nan":
        return float("nan")  # config files: "invalid_value": "nan"
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"invalid_value must be a finite number or NaN, got {value!r}")
    value = float(value)
    if np.isinf(value):
        raise ValueError(
            f"invalid_value must be a finite number or NaN, got {value!r}; "
            "non-finite values are always invalid"
        )
    return value


@dataclass(frozen=True)
class FieldSpec:
    """How a sample field should participate in spatial preprocessing.

    ``invalid_value`` marks pixels without a measurement, e.g. ``0`` in sparse
    lidar depth; non-finite values then count as invalid too. With a
    ``bilinear`` or ``bicubic`` resize the valid values and their support are
    resized separately, so invalid pixels never blend into valid ones, and
    pixels without support get ``invalid_value``. ``invalid_value=NaN`` (or
    ``"nan"`` in a config) suits maps whose holes are NaN: non-finite pixels
    are the invalid ones, and unsupported pixels become NaN instead of
    spreading NaN into their neighbours. ``interpolation="sparse"``
    instead moves every valid pixel to the nearest pixel of the resized grid,
    keeping the nearest surface where several land on one pixel: the smallest
    depth, or the largest disparity. ``invalid_value`` defaults to ``0`` for it.

    ``kind="disparity"`` is stereo disparity in pixels of the field's own
    grid. A resize scales its valid values by the width ratio, since the
    pixels it is measured in change size; a crop leaves them unchanged.
    """

    kind: str = "generic"
    layout: str | None = None
    interpolation: str | None = None
    normalize_vectors: bool | None = None
    threshold: float = 0.5
    reduce: str | None = None
    invalid_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _canonical_kind(self.kind))
        object.__setattr__(self, "layout", _canonical_layout(self.layout))
        object.__setattr__(self, "invalid_value", _check_invalid_value(self.invalid_value))

        if self.interpolation is not None:
            object.__setattr__(self, "interpolation", _check_interpolation(self.interpolation))

        reduce = self.reduce
        if reduce is not None:
            reduce = str(reduce).strip().lower()
            if reduce != "first":
                raise ValueError(
                    f"Unsupported reduction strategy {self.reduce!r}. "
                    "Currently only 'first' is supported."
                )
            object.__setattr__(self, "reduce", reduce)

    @classmethod
    def from_config(cls, cfg: str | Mapping[str, Any] | None) -> "FieldSpec":
        if cfg is None:
            return cls()
        if isinstance(cfg, str):
            return cls(kind=cfg)
        if not isinstance(cfg, Mapping):
            raise TypeError(
                "Field specs must be strings, mappings, or None. "
                f"Got {type(cfg).__name__}."
            )
        return cls(
            kind=str(cfg.get("kind", "generic")),
            layout=cfg.get("layout"),
            interpolation=cfg.get("interpolation"),
            normalize_vectors=cfg.get("normalize_vectors"),
            threshold=float(cfg.get("threshold", 0.5)),
            reduce=cfg.get("reduce"),
            invalid_value=cfg.get("invalid_value"),
        )

    def effective_interpolation(self, value: Any) -> str:
        if self.interpolation is not None:
            return self.interpolation
        # Kinds without a default (e.g. a modality type such as
        # "semantic_segmentation") resize like generic arrays, by dtype.
        default = _DEFAULT_INTERPOLATION.get(self.kind)
        if default is not None:
            return default
        if _is_torch_tensor(value):
            return "nearest" if value.dtype in {torch.bool, torch.int16, torch.int32, torch.int64, torch.uint8} else "bilinear"
        if isinstance(value, np.ndarray):
            return "nearest" if value.dtype.kind in {"b", "i", "u"} else "bilinear"
        return "bilinear"

    def should_normalize_vectors(self) -> bool:
        if self.normalize_vectors is not None:
            return bool(self.normalize_vectors)
        return self.kind == "ray_map"

    def is_passthrough(self) -> bool:
        return self.kind == "passthrough"

    def describe(self, *, include_defaults: bool = True) -> str:
        """Return a concise human-readable description of this field policy."""
        parts = [f"kind={self.kind}"]
        if self.layout is not None:
            parts.append(f"layout={self.layout}")

        interpolation = self.interpolation
        if include_defaults and interpolation is None:
            interpolation = _DEFAULT_INTERPOLATION[self.kind]
        if interpolation is not None:
            parts.append(f"interpolation={interpolation}")

        normalize_vectors = self.normalize_vectors
        if include_defaults and normalize_vectors is None:
            normalize_vectors = self.kind == "ray_map"
        if normalize_vectors is not None:
            parts.append(f"normalize_vectors={bool(normalize_vectors)}")

        if self.kind == "mask" or self.threshold != 0.5:
            parts.append(f"threshold={self.threshold:g}")
        if self.reduce is not None:
            parts.append(f"reduce={self.reduce}")
        if self.invalid_value is not None:
            parts.append(f"invalid_value={self.invalid_value:g}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Resize:
    """Resize the sample to a new ``(height, width)``."""

    size: tuple[int, int]

    def __post_init__(self) -> None:
        height, width = self.size
        if height <= 0 or width <= 0:
            raise ValueError(f"Resize dimensions must be positive, got {self.size}")

    @classmethod
    def from_config(cls, cfg: Any) -> "Resize":
        return cls(size=_parse_size(cfg, context="Resize"))

    def describe(self) -> str:
        height, width = self.size
        return f"resize({height}x{width})"


@dataclass(frozen=True)
class Crop:
    """Crop the sample consistently across all configured fields."""

    size: tuple[int, int]
    anchor: str = "center"
    offset: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        height, width = self.size
        if height <= 0 or width <= 0:
            raise ValueError(f"Crop dimensions must be positive, got {self.size}")
        anchor = str(self.anchor).strip().lower()
        valid_anchors = {
            "center",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        }
        if anchor not in valid_anchors:
            allowed = ", ".join(sorted(valid_anchors))
            raise ValueError(f"Unsupported crop anchor {self.anchor!r}. Available: {allowed}")
        object.__setattr__(self, "anchor", anchor)
        if self.offset is not None and len(self.offset) != 2:
            raise ValueError("Crop offsets must be a pair of [top, left] integers.")

    @classmethod
    def from_config(cls, cfg: Any) -> "Crop":
        if isinstance(cfg, Mapping):
            offset_cfg = cfg.get("offset")
            offset = None
            if offset_cfg is not None:
                if not isinstance(offset_cfg, (list, tuple)) or len(offset_cfg) != 2:
                    raise ValueError("Crop offset must be [top, left].")
                offset = (int(offset_cfg[0]), int(offset_cfg[1]))
            return cls(
                size=_parse_size(cfg, context="Crop"),
                anchor=str(cfg.get("anchor", "center")),
                offset=offset,
            )
        return cls(size=_parse_size(cfg, context="Crop"))

    def describe(self) -> str:
        height, width = self.size
        parts = [f"crop({height}x{width})", f"anchor={self.anchor}"]
        if self.offset is not None:
            parts.append(f"offset={self.offset}")
        return ", ".join(parts)

    def resolve_box(self, source_size: tuple[int, int]) -> tuple[int, int, int, int]:
        source_h, source_w = source_size
        target_h, target_w = self.size
        if target_h > source_h or target_w > source_w:
            raise ValueError(
                f"Cannot crop {self.size} from source size {source_size}."
            )

        if self.offset is not None:
            top, left = self.offset
        elif self.anchor == "center":
            top = (source_h - target_h) // 2
            left = (source_w - target_w) // 2
        elif self.anchor == "top_left":
            top, left = 0, 0
        elif self.anchor == "top_right":
            top, left = 0, source_w - target_w
        elif self.anchor == "bottom_left":
            top, left = source_h - target_h, 0
        else:
            top, left = source_h - target_h, source_w - target_w

        if top < 0 or left < 0 or top + target_h > source_h or left + target_w > source_w:
            raise ValueError(
                f"Crop box {(top, left, target_h, target_w)} is outside source size {source_size}."
            )
        return top, left, target_h, target_w


@dataclass(frozen=True)
class MaskedValueOverride:
    """Set values in one sample field wherever a boolean mask field is true."""

    target_key: str
    mask_key: str
    value: Any
    invert_mask: bool = False
    copy: bool = True
    ignore_missing: bool = False

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MaskedValueOverride":
        if not isinstance(cfg, Mapping):
            raise TypeError(
                "MaskedValueOverride config must be a mapping, "
                f"got {type(cfg).__name__}."
            )
        return cls(
            target_key=str(cfg.get("target_key", cfg.get("target", ""))),
            mask_key=str(cfg.get("mask_key", cfg.get("mask", ""))),
            value=cfg.get("value"),
            invert_mask=bool(cfg.get("invert_mask", False)),
            copy=bool(cfg.get("copy", True)),
            ignore_missing=bool(cfg.get("ignore_missing", False)),
        )

    def __post_init__(self) -> None:
        if not self.target_key:
            raise ValueError("MaskedValueOverride requires target_key.")
        if not self.mask_key:
            raise ValueError("MaskedValueOverride requires mask_key.")

    def describe(self) -> str:
        parts = [
            f"masked_value_override(target={self.target_key}",
            f"mask={self.mask_key}",
            f"value={self.value!r}",
        ]
        if self.invert_mask:
            parts.append("invert_mask=True")
        if not self.copy:
            parts.append("copy=False")
        if self.ignore_missing:
            parts.append("ignore_missing=True")
        return ", ".join(parts) + ")"

    def _missing(self, key: str) -> None:
        if self.ignore_missing:
            return
        raise KeyError(f"MaskedValueOverride missing sample key: {key!r}")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.target_key not in sample:
            self._missing(self.target_key)
            return sample
        if self.mask_key not in sample:
            self._missing(self.mask_key)
            return sample

        target = sample[self.target_key]
        mask = sample[self.mask_key]

        processed = dict(sample) if self.copy else sample

        if _is_torch_tensor(target):
            if _is_torch_tensor(mask):
                mask_tensor = mask.to(device=target.device, dtype=torch.bool)
            else:
                mask_tensor = torch.as_tensor(
                    mask,
                    device=target.device,
                    dtype=torch.bool,
                )
            if self.invert_mask:
                mask_tensor = ~mask_tensor
            if tuple(mask_tensor.shape) != tuple(target.shape):
                mask_tensor = torch.broadcast_to(mask_tensor, target.shape)
            value_tensor = torch.as_tensor(
                self.value,
                device=target.device,
                dtype=target.dtype,
            )
            result = target.clone() if self.copy else target
            result[mask_tensor] = value_tensor
            processed[self.target_key] = result
            return processed

        if isinstance(target, np.ndarray):
            mask_array = np.asarray(mask, dtype=np.bool_)
            if self.invert_mask:
                mask_array = ~mask_array
            if tuple(mask_array.shape) != tuple(target.shape):
                mask_array = np.broadcast_to(mask_array, target.shape)
            result = np.array(target, copy=self.copy)
            result[mask_array] = self.value
            processed[self.target_key] = result
            return processed

        raise TypeError(
            f"MaskedValueOverride target {self.target_key!r} must be a torch.Tensor "
            f"or numpy.ndarray, got {type(target).__name__}."
        )


def infer_field_spec(name: str, value: Any | None = None) -> FieldSpec | None:
    """Best-effort field inference for common modality names."""

    if name in {"id", "full_id", "meta"}:
        return None

    inferred_kind = _infer_kind_from_name(name)
    if value is None:
        return FieldSpec(kind=inferred_kind or "generic") if inferred_kind is not None else None

    if isinstance(value, Mapping):
        return FieldSpec(kind=inferred_kind or "generic") if inferred_kind is not None else None

    if not _is_array_like(value):
        return None

    if value.ndim < 2 or value.ndim > 4:
        return None

    if inferred_kind is None:
        if _is_torch_tensor(value) and value.dtype == torch.bool:
            inferred_kind = "mask"
        elif isinstance(value, np.ndarray) and value.dtype == np.bool_:
            inferred_kind = "mask"
        elif tuple(value.shape[-2:]) == (3, 3) and value.ndim == 2:
            return None
        else:
            inferred_kind = "generic"

    return FieldSpec(kind=inferred_kind)


def _resolve_layout(value: Any, spec: FieldSpec) -> str:
    if spec.layout is not None:
        if len(spec.layout) != value.ndim:
            raise ValueError(
                f"Layout {spec.layout!r} does not match value shape {tuple(value.shape)}."
            )
        return spec.layout

    shape = tuple(int(dim) for dim in value.shape)
    ndim = value.ndim
    if ndim == 2:
        return "HW"
    if ndim == 3:
        if shape[0] == 1:
            return "CHW"
        if shape[-1] == 1:
            return "HWC"
        if spec.kind in {"image", "ray_map"}:
            return "CHW" if _is_torch_tensor(value) else "HWC"
        if shape[0] <= 4 and shape[-1] > 4:
            return "CHW"
        if shape[-1] <= 4 and shape[0] > 4:
            return "HWC"
        return "CHW" if _is_torch_tensor(value) else "HWC"
    if ndim == 4:
        if shape[1] == 1:
            return "NCHW"
        if shape[-1] == 1:
            return "NHWC"
        if shape[1] <= 4 and shape[-1] > 4:
            return "NCHW"
        if shape[-1] <= 4 and shape[1] > 4:
            return "NHWC"
        return "NCHW" if _is_torch_tensor(value) else "NHWC"
    raise ValueError(
        f"Cannot infer layout for value with shape {shape}. "
        "Set FieldSpec.layout explicitly."
    )


def _spatial_size(value: Any, spec: FieldSpec) -> tuple[int, int] | None:
    if value is None or not _is_array_like(value):
        return None
    layout = _resolve_layout(value, spec)
    return int(value.shape[layout.index("H")]), int(value.shape[layout.index("W")])


def _apply_reduction(value: Any, spec: FieldSpec) -> Any:
    if not isinstance(value, Mapping):
        return value
    if spec.reduce is None:
        return value
    if not value:
        return None
    first_key = sorted(value.keys())[0]
    return value[first_key]


def _to_torch_nchw(value: Any, layout: str) -> tuple["torch.Tensor", bool]:
    if torch is None or F is None:
        raise RuntimeError(
            "Torch is required to preprocess torch tensors. "
            "Install the optional `euler-loading[gpu]` dependencies."
        )
    is_torch_input = _is_torch_tensor(value)
    tensor = value if is_torch_input else torch.from_numpy(np.asarray(value))
    if layout == "HW":
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif layout == "CHW":
        tensor = tensor.unsqueeze(0)
    elif layout == "HWC":
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif layout == "NHW":
        tensor = tensor.unsqueeze(1)
    elif layout == "NCHW":
        tensor = tensor
    elif layout == "NHWC":
        tensor = tensor.permute(0, 3, 1, 2)
    else:  # pragma: no cover - guarded by layout validation
        raise ValueError(f"Unsupported layout {layout!r}")
    return tensor, is_torch_input


def _from_torch_nchw(
    tensor: "torch.Tensor",
    *,
    layout: str,
    original: Any,
    original_is_torch: bool,
) -> Any:
    if layout == "HW":
        tensor = tensor.squeeze(0).squeeze(0)
    elif layout == "CHW":
        tensor = tensor.squeeze(0)
    elif layout == "HWC":
        tensor = tensor.squeeze(0).permute(1, 2, 0).contiguous()
    elif layout == "NHW":
        tensor = tensor.squeeze(1)
    elif layout == "NCHW":
        tensor = tensor
    elif layout == "NHWC":
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
    else:  # pragma: no cover - guarded by layout validation
        raise ValueError(f"Unsupported layout {layout!r}")

    if original_is_torch:
        return tensor.to(dtype=original.dtype, device=original.device)
    return tensor.cpu().numpy().astype(original.dtype, copy=False)


def _numpy_to_nchw(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "HW":
        return array[np.newaxis, np.newaxis, :, :]
    if layout == "CHW":
        return array[np.newaxis, :, :, :]
    if layout == "HWC":
        return np.moveaxis(array, -1, 0)[np.newaxis, :, :, :]
    if layout == "NHW":
        return array[:, np.newaxis, :, :]
    if layout == "NCHW":
        return array
    if layout == "NHWC":
        return np.moveaxis(array, -1, 1)
    raise ValueError(f"Unsupported layout {layout!r}")  # pragma: no cover - guarded by layout validation


def _numpy_from_nchw(nchw: np.ndarray, layout: str, dtype: Any) -> np.ndarray:
    if layout == "HW":
        return nchw[0, 0].astype(dtype, copy=False)
    if layout == "CHW":
        return nchw[0].astype(dtype, copy=False)
    if layout == "HWC":
        return np.moveaxis(nchw[0], 0, -1).astype(dtype, copy=False)
    if layout == "NHW":
        return nchw[:, 0].astype(dtype, copy=False)
    if layout == "NCHW":
        return nchw.astype(dtype, copy=False)
    return np.moveaxis(nchw, 1, -1).astype(dtype, copy=False)


def _resize_numpy_fallback(
    value: np.ndarray,
    *,
    layout: str,
    size: tuple[int, int],
    interpolation: str,
) -> np.ndarray:
    target_h, target_w = size
    array = np.asarray(value)
    nchw = _numpy_to_nchw(array, layout)

    resized = np.empty(
        (nchw.shape[0], nchw.shape[1], target_h, target_w),
        dtype=np.float32,
    )
    resample = _PIL_MODES[interpolation]
    for batch_index in range(nchw.shape[0]):
        for channel_index in range(nchw.shape[1]):
            plane = nchw[batch_index, channel_index].astype(np.float32, copy=False)
            image = Image.fromarray(plane, mode="F")
            resized[batch_index, channel_index] = np.asarray(
                image.resize((target_w, target_h), resample=resample),
                dtype=np.float32,
            )

    return _numpy_from_nchw(resized, layout, array.dtype)


def _resize_sparse(
    value: Any,
    *,
    layout: str,
    size: tuple[int, int],
    invalid_value: float,
    keep: str = "min",
) -> Any:
    """Move each valid pixel to the nearest pixel of the resized grid.

    A source pixel centre ``x`` lands at ``(x + 0.5) * scale - 0.5``, the
    ``align_corners=False`` mapping :func:`resize_intrinsics` applies to the
    camera, and is rounded to the nearest target pixel. Where several land on
    one pixel the nearest surface wins: the smallest value with
    ``keep="min"`` (depth), the largest with ``keep="max"`` (disparity).
    Nothing is interpolated or duplicated, so every output value is a source
    measurement; pixels nothing lands on get *invalid_value*, as do invalid
    (non-finite or *invalid_value*) source pixels.
    """
    if keep not in {"min", "max"}:
        raise ValueError(f"keep must be 'min' or 'max', got {keep!r}")
    tensor_input = _is_torch_tensor(value)
    array = value.detach().cpu().numpy() if tensor_input else np.asarray(value)
    if np.isnan(invalid_value) and array.dtype.kind not in "fc":
        raise ValueError(
            f"invalid_value NaN needs a floating-point array, got {array.dtype}"
        )
    nchw = _numpy_to_nchw(array, layout)
    batch, channels, height, width = nchw.shape
    target_h, target_w = size

    valid = np.isfinite(nchw) & (nchw != invalid_value)
    n, c, ys, xs = np.nonzero(valid)
    ty = np.minimum(np.floor((ys + 0.5) * (target_h / height)).astype(np.int64), target_h - 1)
    tx = np.minimum(np.floor((xs + 0.5) * (target_w / width)).astype(np.int64), target_w - 1)
    target = ((n * channels + c) * target_h + ty) * target_w + tx
    if keep == "min":
        flat = np.full(batch * channels * target_h * target_w, np.inf)
        np.minimum.at(flat, target, nchw[valid].astype(np.float64))
    else:
        flat = np.full(batch * channels * target_h * target_w, -np.inf)
        np.maximum.at(flat, target, nchw[valid].astype(np.float64))
    flat[~np.isfinite(flat)] = invalid_value
    resized = _numpy_from_nchw(
        flat.reshape(batch, channels, target_h, target_w), layout, array.dtype
    )
    if tensor_input:
        return torch.from_numpy(np.ascontiguousarray(resized)).to(value.device)
    return resized


def _resize_validity_aware(
    value: Any,
    *,
    layout: str,
    size: tuple[int, int],
    interpolation: str,
    invalid_value: float,
) -> Any:
    """Resize valid values and their support separately; see :class:`FieldSpec`."""
    if _is_torch_tensor(value):
        valid = torch.isfinite(value) & (value != invalid_value)
        clean = torch.where(valid, value, torch.zeros_like(value))
        weight = valid.to(dtype=value.dtype)
    else:
        valid = np.isfinite(value) & (value != invalid_value)
        clean = np.where(valid, value, 0).astype(value.dtype, copy=False)
        weight = valid.astype(value.dtype)
    options = {"layout": layout, "size": size, "interpolation": interpolation,
               "normalize_vectors": False, "threshold": 0.5}
    numerator = _resize_spatial_value(clean, **options)
    support = _resize_spatial_value(weight, **options)
    if _is_torch_tensor(numerator):
        supported = support > 1e-6
        safe = torch.where(supported, support, torch.ones_like(support))
        return torch.where(
            supported, numerator / safe, torch.full_like(numerator, invalid_value)
        )
    supported = support > 1e-6
    out = np.full_like(numerator, invalid_value)
    np.divide(numerator, support, out=out, where=supported)
    return out


def _resize_spatial_value(
    value: Any,
    *,
    layout: str,
    size: tuple[int, int],
    interpolation: str,
    normalize_vectors: bool,
    threshold: float,
    invalid_value: float | None = None,
    sparse_keep: str = "min",
) -> Any:
    if interpolation == "sparse":
        return _resize_sparse(
            value, layout=layout, size=size,
            invalid_value=0.0 if invalid_value is None else invalid_value,
            keep=sparse_keep,
        )
    floating = (
        value.is_floating_point() if _is_torch_tensor(value)
        else isinstance(value, np.ndarray) and value.dtype.kind == "f"
    )
    if invalid_value is not None and interpolation != "nearest" and floating:
        if normalize_vectors:
            raise ValueError("invalid_value cannot be combined with vector normalization.")
        return _resize_validity_aware(
            value, layout=layout, size=size,
            interpolation=interpolation, invalid_value=invalid_value,
        )
    if _is_torch_tensor(value) or torch is not None:
        tensor, original_is_torch = _to_torch_nchw(value, layout)
        original_dtype = tensor.dtype
        work = tensor.to(dtype=torch.float32)
        mode = _TORCH_MODES[interpolation]
        work = F.interpolate(
            work,
            size=size,
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )
        if normalize_vectors:
            work = F.normalize(work, dim=1)
        if original_dtype == torch.bool:
            work = work > threshold
        elif original_dtype in {torch.int16, torch.int32, torch.int64, torch.uint8}:
            work = work.round().to(dtype=original_dtype)
        else:
            work = work.to(dtype=original_dtype)
        return _from_torch_nchw(
            work,
            layout=layout,
            original=value,
            original_is_torch=original_is_torch,
        )

    if not isinstance(value, np.ndarray):
        return value

    original_dtype = value.dtype
    resized = _resize_numpy_fallback(
        value,
        layout=layout,
        size=size,
        interpolation=interpolation,
    )
    if normalize_vectors:
        if layout in {"HW", "NHW"}:
            raise ValueError("Vector normalization requires a channel dimension.")
        axis = {"CHW": 0, "HWC": -1, "NCHW": 1, "NHWC": -1}[layout]
        norms = np.linalg.norm(resized, axis=axis, keepdims=True)
        norms = np.clip(norms, a_min=1.0e-8, a_max=None)
        resized = resized / norms
    if original_dtype == np.bool_:
        return resized > threshold
    if original_dtype.kind in {"i", "u"}:
        return np.rint(resized).astype(original_dtype)
    return resized.astype(original_dtype, copy=False)


def resize_raster(
    value: np.ndarray | "torch.Tensor",
    *,
    layout: str,
    size: tuple[int, int],
    interpolation: str = "bilinear",
    invalid_value: float | None = None,
) -> np.ndarray | "torch.Tensor":
    """Resize one raster to ``size = (height, width)``, as :class:`Resize` resizes sample fields.

    *layout* is ``"HW"``, ``"HWC"`` or ``"CHW"`` (or a batched ``N...`` form).
    *interpolation* is ``"nearest"``, ``"bilinear"``, ``"bicubic"`` or
    ``"sparse"``, all in the pixel-centre (``align_corners=False``) convention
    :func:`resize_intrinsics` assumes. ``nearest`` samples at pixel centres
    with torch and Pillow alike. Integer values are rounded and boolean values
    thresholded at 0.5 after interpolation; the dtype and the container (array
    or tensor) are kept. *invalid_value* marks pixels without a measurement;
    see :class:`FieldSpec` for how it and ``"sparse"`` treat them.
    """
    canonical = _canonical_layout(layout)
    if canonical is None:
        raise ValueError("layout is required")
    mode = _check_interpolation(interpolation)
    invalid_value = _check_invalid_value(invalid_value)
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Resize dimensions must be positive, got {size}")
    if not _is_array_like(value):
        raise TypeError(f"Expected a NumPy array or torch tensor, got {type(value).__name__}")
    return _resize_spatial_value(
        value,
        layout=canonical,
        size=(height, width),
        interpolation=mode,
        normalize_vectors=False,
        threshold=0.5,
        invalid_value=invalid_value,
    )


def resize_intrinsics(
    intrinsics: np.ndarray | "torch.Tensor",
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray | "torch.Tensor":
    """Scale pinhole intrinsics for an ``align_corners=False`` image resize."""

    from .geometry import resize_matrix, transform_intrinsics
    return transform_intrinsics(intrinsics, resize_matrix(source_size, target_size))


def crop_intrinsics(
    intrinsics: np.ndarray | "torch.Tensor",
    *,
    top: int,
    left: int,
) -> np.ndarray | "torch.Tensor":
    """Adjust pinhole intrinsics after cropping pixels from the image border."""

    from .geometry import crop_matrix, transform_intrinsics
    return transform_intrinsics(intrinsics, crop_matrix(top, left))


def _crop_spatial_value(
    value: Any,
    *,
    layout: str,
    top: int,
    left: int,
    size: tuple[int, int],
) -> Any:
    target_h, target_w = size
    h_axis = layout.index("H")
    w_axis = layout.index("W")
    slices = [slice(None)] * value.ndim
    slices[h_axis] = slice(top, top + target_h)
    slices[w_axis] = slice(left, left + target_w)
    return value[tuple(slices)]


def _dataset_lookup(dataset: Any, accessor: str, name: str) -> Any:
    lookup = getattr(dataset, accessor, None)
    if not callable(lookup):
        return None
    try:
        return lookup(name)
    except Exception:
        return None


#: Kinds whose pixels are measurements, resized without blending in invalid ones.
_MEASURED_KINDS = ("depth", "disparity")


def _bind_measured_map(dataset: Any, name: str, spec: FieldSpec | None) -> FieldSpec | None:
    """Return *spec* with the kind and invalid-pixel handling a measured map declares.

    The field kind comes from *spec*, else the loader's ``@modality_meta``
    type, else the field name. Only depth and disparity fields are changed:

    - ``invalid`` and ``sparse`` select the resize, where *spec* sets neither
      ``interpolation`` nor ``invalid_value``: ``sparse: true`` moves each
      measurement to its nearest target pixel (``interpolation="sparse"``); an
      ``invalid.sentinel`` alone selects a validity-aware bilinear resize, as
      does ``invalid.non_finite: true`` with no sentinel, which marks the
      holes with NaN (``invalid_value=NaN``). Each key is read from the modality's head meta, else
      from the meta of the loader's annotation, which states what the loader
      always returns (DrivingStereo's maps are sparse with ``0`` invalid);
    - a disparity field is bound even when its head declares neither, because
      its values rescale with a resize; see :class:`FieldSpec`.
    """
    candidate = spec
    annotation = _dataset_lookup(dataset, "get_loader_meta", name)
    if not isinstance(annotation, Mapping):
        annotation = {}
    if candidate is None:
        loader_type = annotation.get("type")
        kind = _canonical_kind(loader_type or _infer_kind_from_name(name))
        if kind not in _MEASURED_KINDS:
            return None
        candidate = FieldSpec(kind=kind)
    if (
        candidate.kind not in _MEASURED_KINDS
        or candidate.interpolation is not None
        or candidate.invalid_value is not None
    ):
        return spec
    head = _dataset_lookup(dataset, "get_modality_metadata", name)
    head = head if isinstance(head, Mapping) else {}
    loader_meta = annotation.get("meta")
    loader_meta = loader_meta if isinstance(loader_meta, Mapping) else {}

    def declared(key: str) -> Any:
        return head[key] if key in head else loader_meta.get(key)

    invalid = declared("invalid")
    sentinel = invalid.get("sentinel") if isinstance(invalid, Mapping) else None
    nan_holes = isinstance(invalid, Mapping) and invalid.get("non_finite") is True
    sparse = declared("sparse") is True
    if sentinel is None and not sparse and not nan_holes:
        return candidate if candidate.kind == "disparity" else spec
    if sentinel is not None:
        invalid_value = sentinel
    else:
        # NaN-marked holes, else the sparse default sentinel 0.
        invalid_value = float("nan") if nan_holes else 0.0
    return replace(
        candidate,
        interpolation="sparse" if sparse else None,
        invalid_value=invalid_value,
    )


def _scale_valid(value: Any, factor: float, invalid_value: float | None) -> Any:
    """Multiply the valid (finite, non-sentinel) pixels of *value* by *factor*."""
    if factor == 1:
        return value
    if _is_torch_tensor(value):
        valid = torch.isfinite(value)
        if invalid_value is not None:
            valid &= value != invalid_value
        return torch.where(valid, value * factor, value)
    array = np.asarray(value)
    valid = np.isfinite(array)
    if invalid_value is not None:
        valid &= array != invalid_value
    return np.where(valid, array * factor, array).astype(array.dtype, copy=False)


@dataclass
class SamplePreprocessor:
    """Apply coordinated spatial preprocessing to sample dictionaries."""

    operations: list[Resize | Crop] = field(default_factory=list)
    field_specs: dict[str, FieldSpec] = field(default_factory=dict)
    reference_field: str | None = None
    infer_fields: bool = True
    _bound_field_specs: dict[str, FieldSpec] = field(default_factory=dict, init=False, repr=False)
    _authoring_config: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def export_descriptor(self, **bindings: Any) -> Any:
        """Export a frozen Phase 1 plan; see transform_descriptors.export_preprocessor.

        Requires declared source/profile/plane bindings and an explicit execution
        profile. The legacy callable behavior is independent of this opt-in plan.
        """
        from .transform_descriptors import export_preprocessor

        return export_preprocessor(self, **bindings)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "SamplePreprocessor":
        cfg = cfg or {}
        if not isinstance(cfg, Mapping):
            raise TypeError(
                "Preprocessing config must be a mapping or None, "
                f"got {type(cfg).__name__}."
            )

        field_cfg = cfg.get("fields", cfg.get("field_specs", {}))
        if field_cfg is None:
            field_specs: dict[str, FieldSpec] = {}
        elif not isinstance(field_cfg, Mapping):
            raise TypeError("`fields` must be a mapping of field name to field spec.")
        else:
            field_specs = {
                str(name): FieldSpec.from_config(spec)
                for name, spec in field_cfg.items()
            }

        operations_cfg = cfg.get("operations")
        operations: list[Resize | Crop] = []
        if operations_cfg is None:
            if cfg.get("resize") is not None:
                operations.append(Resize.from_config(cfg["resize"]))
            if cfg.get("crop") is not None:
                operations.append(Crop.from_config(cfg["crop"]))
        else:
            if not isinstance(operations_cfg, list):
                raise TypeError("`operations` must be a list of preprocessing steps.")
            for entry in operations_cfg:
                if not isinstance(entry, Mapping):
                    raise TypeError("Each preprocessing operation must be a mapping.")
                op_type = str(entry.get("type", "")).strip().lower()
                if op_type == "resize":
                    operations.append(Resize.from_config(entry))
                elif op_type == "crop":
                    operations.append(Crop.from_config(entry))
                else:
                    raise ValueError(
                        f"Unsupported preprocessing operation {op_type!r}. "
                        "Available operations: resize, crop"
                    )

        result = cls(
            operations=operations,
            field_specs=field_specs,
            reference_field=cfg.get("reference_field"),
            infer_fields=bool(cfg.get("infer_fields", True)),
        )
        result._authoring_config = deepcopy(dict(cfg))
        return result

    def bind_to_dataset(self, dataset: Any) -> None:
        """Optionally enrich field specs from dataset modality metadata.

        A modality's ``modality_type`` sets its field kind. A depth or
        disparity field whose dataset head declares invalid pixels is resized
        without blending them in: ``sparse: true`` selects
        ``interpolation="sparse"``, and ``invalid.sentinel`` alone a
        validity-aware bilinear resize. A disparity field is recognised from
        its loader's type even without those, so a resize rescales its values.
        Explicit ``field_specs`` always win.
        """

        bound_specs = dict(self.field_specs)

        def bind_group(modalities: Mapping[str, Any]) -> None:
            for name, modality in modalities.items():
                if name in bound_specs:
                    continue
                modality_type = getattr(modality, "modality_type", None)
                spec = FieldSpec(kind=str(modality_type)) if modality_type else None
                spec = _bind_measured_map(dataset, name, spec)
                if spec is not None:
                    bound_specs[name] = spec

        bind_group(getattr(dataset, "_modalities", {}))
        bind_group(getattr(dataset, "_hierarchical_modalities", {}))
        self._bound_field_specs = bound_specs

    def describe(self) -> str:
        """Return a concise summary suitable for initialization logging."""
        operations = ", ".join(
            operation.describe() if hasattr(operation, "describe") else repr(operation)
            for operation in self.operations
        ) or "none"

        active_specs = self._bound_field_specs or self.field_specs
        if active_specs:
            field_parts = [
                f"{name}: {spec.describe()}"
                for name, spec in sorted(active_specs.items())
            ]
            fields_text = "; ".join(field_parts)
        elif self.infer_fields:
            fields_text = "inferred from modality_type/name heuristics"
        else:
            fields_text = "none"

        reference = self.reference_field or "auto"
        return (
            "SamplePreprocessor("
            f"operations=[{operations}], "
            f"reference_field={reference}, "
            f"infer_fields={self.infer_fields}, "
            f"fields={fields_text})"
        )

    def _resolve_field_spec(self, key: str, value: Any) -> FieldSpec | None:
        if key in self._bound_field_specs:
            return self._bound_field_specs[key]
        if key in self.field_specs:
            return self.field_specs[key]
        if not self.infer_fields:
            return None
        return infer_field_spec(key, value)

    def _infer_reference_size(
        self,
        sample: Mapping[str, Any],
        specs: Mapping[str, FieldSpec],
    ) -> tuple[int, int] | None:
        if self.reference_field is not None:
            value = sample.get(self.reference_field)
            if value is None:
                return None
            spec = specs.get(self.reference_field) or self._resolve_field_spec(
                self.reference_field, value
            )
            if spec is None:
                return None
            size = _spatial_size(value, spec)
            if size is None:
                raise ValueError(
                    f"Reference field {self.reference_field!r} is not spatial."
                )
            return size

        for key, spec in specs.items():
            size = _spatial_size(sample.get(key), spec)
            if size is not None:
                return size
        return None

    def _apply_operation(
        self,
        *,
        value: Any,
        spec: FieldSpec,
        operation: Resize | Crop,
        reference_size: tuple[int, int],
    ) -> Any:
        if value is None or spec.is_passthrough():
            return value

        if spec.kind == "intrinsics":
            if isinstance(operation, Resize):
                return resize_intrinsics(
                    value,
                    source_size=reference_size,
                    target_size=operation.size,
                )
            top, left, _, _ = operation.resolve_box(reference_size)
            return crop_intrinsics(value, top=top, left=left)

        if not _is_array_like(value):
            return value

        layout = _resolve_layout(value, spec)
        value_size = _spatial_size(value, spec)
        if value_size is None:
            return value

        if isinstance(operation, Resize):
            if value_size == operation.size:
                return value
            disparity = spec.kind == "disparity"
            resized = _resize_spatial_value(
                value,
                layout=layout,
                size=operation.size,
                interpolation=spec.effective_interpolation(value),
                normalize_vectors=spec.should_normalize_vectors(),
                threshold=spec.threshold,
                invalid_value=spec.invalid_value,
                # The nearest surface has the largest disparity.
                sparse_keep="max" if disparity else "min",
            )
            if disparity:
                # Disparity is measured in pixels of this grid, which the
                # resize makes wider or narrower.
                resized = _scale_valid(
                    resized, operation.size[1] / value_size[1], spec.invalid_value
                )
            return resized

        top, left, target_h, target_w = operation.resolve_box(reference_size)
        if value_size == (target_h, target_w):
            return value
        return _crop_spatial_value(
            value,
            layout=layout,
            top=top,
            left=left,
            size=(target_h, target_w),
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.operations and not any(spec.reduce for spec in self.field_specs.values()):
            return sample

        processed = dict(sample)
        specs: dict[str, FieldSpec] = {}

        for key, value in sample.items():
            spec = self._resolve_field_spec(key, value)
            if spec is None:
                continue
            processed[key] = _apply_reduction(value, spec)
            specs[key] = spec

        for operation in self.operations:
            reference_size = self._infer_reference_size(processed, specs)
            if reference_size is None:
                continue
            for key, spec in specs.items():
                processed[key] = self._apply_operation(
                    value=processed.get(key),
                    spec=spec,
                    operation=operation,
                    reference_size=reference_size,
                )

        return processed
