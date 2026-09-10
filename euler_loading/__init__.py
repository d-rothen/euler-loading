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
    MaskedValueOverride,
    Resize,
    SamplePreprocessor,
    crop_intrinsics,
    infer_field_spec,
    resize_intrinsics,
)
from .transform_descriptors import (
    ResolvedOperation,
    ResolvedTransform,
    SerializableTransform,
    execution_profile,
    export_preprocessor,
    resolve_transform_descriptor,
)

__all__ = [
    'CalibrationView',
    'DatasetCapture',
    'execute_with_receipt',
    'replay_from_arrays',
    'output_encoding',
    'create_materialized_writer',
    'write_materialized',

    "ResolvedOperation",
    "ResolvedTransform",
    "SerializableTransform",
    "execution_profile",
    "export_preprocessor",
    "resolve_transform_descriptor",
    "DenseDepthCodec",
    "DenseDepthLoader",
    "DenseDepthWriter",
    "FileRecord",
    "Crop",
    "FieldSpec",
    "MaskedValueOverride",
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

from .materialization import create_materialized_writer, write_materialized
from .output_encoding import output_encoding
from .receipts import (
    CalibrationView,
    DatasetCapture,
    execute_with_receipt,
    replay_from_arrays,
)
