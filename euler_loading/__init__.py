"""euler-loading: Multi-modal PyTorch dataloader using ds-crawler indices."""

from .dataset import Modality, MultiModalDataset
from .indexing import FileRecord
from .loaders.contracts import DenseDepthLoader

__all__ = ["DenseDepthLoader", "FileRecord", "Modality", "MultiModalDataset"]
