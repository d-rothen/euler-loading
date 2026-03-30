"""euler-loading: Multi-modal PyTorch dataloader using ds-crawler indices."""

from . import _dataset_contract  # noqa: F401
from ._resolution import resolve_loader_module, resolve_writer_module
from ._writing import create_dataset_writer_from_index
from .dataset import Modality, MultiModalDataset
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
