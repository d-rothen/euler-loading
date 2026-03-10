"""euler-loading: Multi-modal PyTorch dataloader using ds-crawler indices."""

from .dataset import (
    create_dataset_writer_from_index,
    Modality,
    MultiModalDataset,
    resolve_loader_module,
    resolve_writer_module,
)
from .indexing import FileRecord
from .loaders.contracts import DenseDepthCodec, DenseDepthLoader, DenseDepthWriter

__all__ = [
    "DenseDepthCodec",
    "DenseDepthLoader",
    "DenseDepthWriter",
    "FileRecord",
    "Modality",
    "MultiModalDataset",
    "create_dataset_writer_from_index",
    "resolve_loader_module",
    "resolve_writer_module",
]
