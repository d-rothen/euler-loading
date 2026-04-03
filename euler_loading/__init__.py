"""euler-loading: Multi-modal PyTorch dataloader using ds-crawler indices."""

from . import _dataset_contract  # noqa: F401
from ._resolution import resolve_loader_module, resolve_writer_module
from ._writing import create_dataset_writer_from_index
from .dataset import Modality, MultiModalDataset
from .indexing import FileRecord
from .loaders.contracts import DenseDepthCodec, DenseDepthLoader, DenseDepthWriter
from .preprocessing import (
    Crop,
    FieldSpec,
    Resize,
    SamplePreprocessor,
    crop_intrinsics,
    infer_field_spec,
    resize_intrinsics,
)

__all__ = [
    "DenseDepthCodec",
    "DenseDepthLoader",
    "DenseDepthWriter",
    "FileRecord",
    "Crop",
    "FieldSpec",
    "Modality",
    "MultiModalDataset",
    "Resize",
    "SamplePreprocessor",
    "create_dataset_writer_from_index",
    "crop_intrinsics",
    "infer_field_spec",
    "resolve_loader_module",
    "resolve_writer_module",
    "resize_intrinsics",
]
