"""Example: load a VKITTI2 multi-modal sample and print it as JSON."""

from __future__ import annotations

import json

import torch
import numpy as np
from PIL import Image

from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu import vkitti2 as vkitti2_cpu
from euler_loading.loaders.gpu import vkitti2 as vkitti2_gpu

# ---------------------------------------------------------------------------
# Pre-configured VKITTI2 paths
# ---------------------------------------------------------------------------

VKITTI2_ROOT = "/Volumes/Volume/Datasets/vkitti2"

RGB_PATH = f"{VKITTI2_ROOT}/vkitti_2.0.3_rgb"
DEPTH_PATH = f"{VKITTI2_ROOT}/vkitti_2.0.3_depth"
CLASS_SEG_PATH = f"{VKITTI2_ROOT}/vkitti_2.0.3_classSegmentation"
TEXTGT_INTRINSICS_PATH = f"{VKITTI2_ROOT}/vkitti_2.0.3_textgt"

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

cpu_dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(RGB_PATH, loader=vkitti2_cpu.rgb),
        "depth": Modality(DEPTH_PATH, loader=vkitti2_cpu.depth),
        "classSegmentation": Modality(CLASS_SEG_PATH, loader=vkitti2_cpu.class_segmentation),
    },
    hierarchical_modalities={
        "textgt_intrinsics": Modality(
            TEXTGT_INTRINSICS_PATH, loader=vkitti2_cpu.read_intrinsics
        ),
    },
)

gpu_dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(RGB_PATH, loader=vkitti2_gpu.rgb),
        "depth": Modality(DEPTH_PATH, loader=vkitti2_gpu.depth),
        "classSegmentation": Modality(CLASS_SEG_PATH, loader=vkitti2_gpu.class_segmentation),
    },
    hierarchical_modalities={
        "textgt_intrinsics": Modality(
            TEXTGT_INTRINSICS_PATH, loader=vkitti2_gpu.read_intrinsics
        ),
    },
)

# ---------------------------------------------------------------------------
# Print the first sample from each dataset as formatted JSON
# ---------------------------------------------------------------------------


def _make_serializable(obj: object) -> object:
    """Replace non-JSON-serializable objects with descriptive strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Image.Image):
        return f"<PIL.Image mode={obj.mode} size={obj.size}>"
    if isinstance(obj, torch.Tensor):
        return f"<torch.Tensor shape={tuple(obj.shape)} dtype={obj.dtype}>"
    if isinstance(obj, np.ndarray):
        return f"<numpy.ndarray shape={obj.shape} dtype={obj.dtype}>"
    return obj


for name, dataset, output_path in [
    ("CPU", cpu_dataset, "vkitti_cpu_example_output.json"),
    ("GPU", gpu_dataset, "vkitti_gpu_example_output.json"),
]:
    sample = dataset[0]
    serialized = json.dumps(_make_serializable(sample), indent=2)
    print(f"--- {name} loader ---")
    print(serialized)
    print()
    with open(output_path, "w") as f:
        f.write(serialized)
