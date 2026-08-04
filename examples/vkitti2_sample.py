"""Load one Virtual KITTI 2 sample and print its structure.

Usage::

    export VKITTI2_ROOT=/path/to/vkitti2
    python examples/vkitti2_sample.py          # torch tensors (CHW)
    python examples/vkitti2_sample.py --cpu    # numpy arrays  (HWC)

``VKITTI2_ROOT`` must contain the standard VKITTI 2 distribution directories,
each already indexed by ds-crawler::

    $VKITTI2_ROOT/
      vkitti_2.0.3_rgb/
      vkitti_2.0.3_depth/
      vkitti_2.0.3_classSegmentation/
      vkitti_2.0.3_textgt/
"""

from __future__ import annotations

import argparse
import os
import sys

from euler_loading import Modality, MultiModalDataset


def describe(value: object) -> str:
    """Render a loaded modality value as a one-line shape/dtype summary."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__} shape={tuple(shape)} dtype={value.dtype}"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {describe(v)}" for k, v in value.items())
        return f"{{{inner}}}"
    return f"{type(value).__name__} {value!r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load one Virtual KITTI 2 sample and print its structure."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("VKITTI2_ROOT"),
        help="VKITTI 2 dataset root (defaults to $VKITTI2_ROOT).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use the numpy (CPU) loaders instead of the torch (GPU) ones.",
    )
    args = parser.parse_args()

    if not args.root:
        parser.error("set VKITTI2_ROOT or pass --root /path/to/vkitti2")

    # Picking the variant explicitly is what makes this example CPU or GPU.
    # Leaving `loader=None` instead would let euler-loading resolve the loader
    # from the ds-crawler index, which defaults to the GPU (torch) variant.
    if args.cpu:
        from euler_loading.loaders.cpu import vkitti2
    else:
        from euler_loading.loaders.gpu import vkitti2

    root = args.root.rstrip("/")
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(f"{root}/vkitti_2.0.3_rgb", loader=vkitti2.rgb),
            "depth": Modality(f"{root}/vkitti_2.0.3_depth", loader=vkitti2.depth),
            "classSegmentation": Modality(
                f"{root}/vkitti_2.0.3_classSegmentation",
                loader=vkitti2.class_segmentation,
            ),
        },
        hierarchical_modalities={
            "intrinsics": Modality(
                f"{root}/vkitti_2.0.3_textgt",
                loader=vkitti2.read_intrinsics,
            ),
        },
    )

    print(f"variant:  {'cpu (numpy)' if args.cpu else 'gpu (torch)'}")
    print(f"samples:  {len(dataset)}")

    sample = dataset[0]
    print(f"id:       {sample['id']}")
    print(f"full_id:  {sample['full_id']}")
    print("modalities:")
    for key, value in sample.items():
        if key in {"id", "full_id", "meta", "attributes"}:
            continue
        print(f"  {key}: {describe(value)}")

    print("source files:")
    for name, entry in sample["meta"].items():
        print(f"  {name}: {entry['path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
