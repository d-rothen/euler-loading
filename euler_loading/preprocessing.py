from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
    "Resize",
    "Crop",
    "SamplePreprocessor",
    "infer_field_spec",
    "resize_intrinsics",
    "crop_intrinsics",
]


_PIL_RESAMPLING = getattr(Image, "Resampling", Image)
_PIL_MODES = {
    "nearest": _PIL_RESAMPLING.NEAREST,
    "bilinear": _PIL_RESAMPLING.BILINEAR,
    "bicubic": _PIL_RESAMPLING.BICUBIC,
}
_FIELD_KIND_ALIASES = {
    "auto": "generic",
    "camera_rays": "ray_map",
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
}
_DEFAULT_INTERPOLATION = {
    "depth": "bilinear",
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
    if "depth" in lowered:
        return "depth"
    return None


@dataclass(frozen=True)
class FieldSpec:
    """How a sample field should participate in spatial preprocessing."""

    kind: str = "generic"
    layout: str | None = None
    interpolation: str | None = None
    normalize_vectors: bool | None = None
    threshold: float = 0.5
    reduce: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _canonical_kind(self.kind))
        object.__setattr__(self, "layout", _canonical_layout(self.layout))

        interpolation = self.interpolation
        if interpolation is not None:
            interpolation = str(interpolation).strip().lower()
            if interpolation not in _PIL_MODES:
                available = ", ".join(sorted(_PIL_MODES))
                raise ValueError(
                    f"Unsupported interpolation {self.interpolation!r}. "
                    f"Available interpolations: {available}"
                )
            object.__setattr__(self, "interpolation", interpolation)

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
        )

    def effective_interpolation(self, value: Any) -> str:
        if self.interpolation is not None:
            return self.interpolation
        default = _DEFAULT_INTERPOLATION[self.kind]
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


def _resize_numpy_fallback(
    value: np.ndarray,
    *,
    layout: str,
    size: tuple[int, int],
    interpolation: str,
) -> np.ndarray:
    target_h, target_w = size
    array = np.asarray(value)
    if layout == "HW":
        nchw = array[np.newaxis, np.newaxis, :, :]
    elif layout == "CHW":
        nchw = array[np.newaxis, :, :, :]
    elif layout == "HWC":
        nchw = np.moveaxis(array, -1, 0)[np.newaxis, :, :, :]
    elif layout == "NHW":
        nchw = array[:, np.newaxis, :, :]
    elif layout == "NCHW":
        nchw = array
    elif layout == "NHWC":
        nchw = np.moveaxis(array, -1, 1)
    else:  # pragma: no cover - guarded by layout validation
        raise ValueError(f"Unsupported layout {layout!r}")

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

    if layout == "HW":
        return resized[0, 0].astype(array.dtype, copy=False)
    if layout == "CHW":
        return resized[0].astype(array.dtype, copy=False)
    if layout == "HWC":
        return np.moveaxis(resized[0], 0, -1).astype(array.dtype, copy=False)
    if layout == "NHW":
        return resized[:, 0].astype(array.dtype, copy=False)
    if layout == "NCHW":
        return resized.astype(array.dtype, copy=False)
    return np.moveaxis(resized, 1, -1).astype(array.dtype, copy=False)


def _resize_spatial_value(
    value: Any,
    *,
    layout: str,
    size: tuple[int, int],
    interpolation: str,
    normalize_vectors: bool,
    threshold: float,
) -> Any:
    if _is_torch_tensor(value) or torch is not None:
        tensor, original_is_torch = _to_torch_nchw(value, layout)
        original_dtype = tensor.dtype
        work = tensor.to(dtype=torch.float32)
        work = F.interpolate(
            work,
            size=size,
            mode=interpolation,
            align_corners=False if interpolation != "nearest" else None,
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


def resize_intrinsics(
    intrinsics: np.ndarray | "torch.Tensor",
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray | "torch.Tensor":
    """Scale pinhole intrinsics for an ``align_corners=False`` image resize."""

    source_h, source_w = source_size
    target_h, target_w = target_size
    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)

    if _is_torch_tensor(intrinsics):
        K = intrinsics.clone()
    else:
        K = np.array(intrinsics, copy=True)

    if K.ndim == 2:
        if K.shape != (3, 3):
            raise ValueError(f"Expected intrinsics with shape (3, 3), got {tuple(K.shape)}")
        K[0, 0] = K[0, 0] * scale_x
        K[1, 1] = K[1, 1] * scale_y
        K[0, 2] = (K[0, 2] + 0.5) * scale_x - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * scale_y - 0.5
        return K

    if K.ndim == 3 and K.shape[-2:] == (3, 3):
        K[..., 0, 0] = K[..., 0, 0] * scale_x
        K[..., 1, 1] = K[..., 1, 1] * scale_y
        K[..., 0, 2] = (K[..., 0, 2] + 0.5) * scale_x - 0.5
        K[..., 1, 2] = (K[..., 1, 2] + 0.5) * scale_y - 0.5
        return K

    raise ValueError(f"Expected intrinsics with shape (3, 3) or (N, 3, 3), got {tuple(K.shape)}")


def crop_intrinsics(
    intrinsics: np.ndarray | "torch.Tensor",
    *,
    top: int,
    left: int,
) -> np.ndarray | "torch.Tensor":
    """Adjust pinhole intrinsics after cropping pixels from the image border."""

    if _is_torch_tensor(intrinsics):
        K = intrinsics.clone()
    else:
        K = np.array(intrinsics, copy=True)

    if K.ndim == 2:
        if K.shape != (3, 3):
            raise ValueError(f"Expected intrinsics with shape (3, 3), got {tuple(K.shape)}")
        K[0, 2] = K[0, 2] - float(left)
        K[1, 2] = K[1, 2] - float(top)
        return K

    if K.ndim == 3 and K.shape[-2:] == (3, 3):
        K[..., 0, 2] = K[..., 0, 2] - float(left)
        K[..., 1, 2] = K[..., 1, 2] - float(top)
        return K

    raise ValueError(f"Expected intrinsics with shape (3, 3) or (N, 3, 3), got {tuple(K.shape)}")


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


@dataclass
class SamplePreprocessor:
    """Apply coordinated spatial preprocessing to sample dictionaries."""

    operations: list[Resize | Crop] = field(default_factory=list)
    field_specs: dict[str, FieldSpec] = field(default_factory=dict)
    reference_field: str | None = None
    infer_fields: bool = True
    _bound_field_specs: dict[str, FieldSpec] = field(default_factory=dict, init=False, repr=False)

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

        return cls(
            operations=operations,
            field_specs=field_specs,
            reference_field=cfg.get("reference_field"),
            infer_fields=bool(cfg.get("infer_fields", True)),
        )

    def bind_to_dataset(self, dataset: Any) -> None:
        """Optionally enrich field specs from dataset modality metadata."""

        bound_specs = dict(self.field_specs)

        def bind_group(modalities: Mapping[str, Any]) -> None:
            for name, modality in modalities.items():
                if name in bound_specs:
                    continue
                modality_type = getattr(modality, "modality_type", None)
                if not modality_type:
                    continue
                bound_specs[name] = FieldSpec(kind=str(modality_type))

        bind_group(getattr(dataset, "_modalities", {}))
        bind_group(getattr(dataset, "_hierarchical_modalities", {}))
        self._bound_field_specs = bound_specs

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
            return _resize_spatial_value(
                value,
                layout=layout,
                size=operation.size,
                interpolation=spec.effective_interpolation(value),
                normalize_vectors=spec.should_normalize_vectors(),
                threshold=spec.threshold,
            )

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
