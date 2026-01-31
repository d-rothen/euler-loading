"""Loader functions for the Virtual KITTI 2 dataset.

Each function follows the ``Callable[[str], Any]`` signature expected by
:class:`~euler_loading.Modality`.  Image loaders use lazy PIL imports so the
module can be introspected without requiring Pillow at import time.

Usage::

    from euler_loading.loaders import vkitti2
    from euler_loading import Modality

    Modality("/data/vkitti2/rgb",   loader=vkitti2.rgb)
    Modality("/data/vkitti2/depth", loader=vkitti2.depth)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Image modality loaders
# ---------------------------------------------------------------------------


def rgb(path: str) -> Any:
    """Load an RGB image (PIL ``Image``, mode ``RGB``)."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def depth(path: str) -> Any:
    """Load a VKITTI2 depth map as a float32 numpy array in **metres**.

    VKITTI2 stores depth as 16-bit PNG where each pixel value represents
    depth in centimetres.  This loader converts to metres
    (``value / 100``).
    """
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path), dtype=np.float32) / 100.0


def class_segmentation(path: str) -> Any:
    """Load an RGB-encoded class-segmentation mask."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def instance_segmentation(path: str) -> Any:
    """Load an RGB-encoded instance-segmentation mask."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def scene_flow(path: str) -> Any:
    """Load an optical / scene-flow map."""
    from PIL import Image

    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Text ground-truth readers (calibration)
# ---------------------------------------------------------------------------


def read_intrinsics(path: str) -> dict[str, float]:
    """Parse a VKITTI2 intrinsics text file.

    The file has the header ``frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]``.
    Intrinsics are constant across frames in VKITTI2, so only the first row
    is used.  Returns a dict with keys ``fx``, ``fy``, ``cx``, ``cy``, and
    ``s`` (skew, always 0 in VKITTI2).
    """
    import numpy as np

    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    #In VKITTI2, intrinsics are constant across frames, so just read the first row
    return {
        "fx": float(data[0][2]),
        "fy": float(data[0][3]),
        "cx": float(data[0][4]),
        "cy": float(data[0][5]),
        "s": 0.0,
    }


def read_extrinsics(path: str) -> Any:
    """Parse a VKITTI2 extrinsics text file into a numpy array.

    Same conventions as :func:`read_intrinsics`.
    """
    import numpy as np

    return np.loadtxt(path)
