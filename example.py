"""Example: load a VKITTI2 multi-modal sample and print it as JSON."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders import vkitti2

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

dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(RGB_PATH, loader=vkitti2.rgb),
        "depth": Modality(DEPTH_PATH, loader=vkitti2.depth),
        "classSegmentation": Modality(CLASS_SEG_PATH, loader=vkitti2.class_segmentation),
    },
    hierarchical_modalities={
        "textgt_intrinsics": Modality(
            TEXTGT_INTRINSICS_PATH, loader=vkitti2.read_intrinsics
        ),
    },
)

# ---------------------------------------------------------------------------
# Print the first sample as formatted JSON
# ---------------------------------------------------------------------------


def _make_serializable(obj: object) -> object:
    """Replace non-JSON-serializable objects with descriptive strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Image.Image):
        return f"<PIL.Image mode={obj.mode} size={obj.size}>"
    if isinstance(obj, np.ndarray):
        return f"<numpy.ndarray shape={obj.shape} dtype={obj.dtype}>"
    return obj


sample = dataset[0]
first_item = json.dumps(_make_serializable(sample), indent=2)
print(first_item)

with open("example_output.json", "w") as f:
    f.write(first_item)
