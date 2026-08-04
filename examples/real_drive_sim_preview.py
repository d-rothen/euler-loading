"""Display the first few RGB frames of a Real Drive Sim modality.

Usage::

    export REAL_DRIVE_SIM_RGB=/path/to/real-drive-sim/rgb
    python examples/real_drive_sim_preview.py --count 3

Requires ``matplotlib``, which is not a dependency of euler-loading::

    pip install matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys

from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.gpu import real_drive_sim


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Display the first few RGB frames of a Real Drive Sim modality."
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("REAL_DRIVE_SIM_RGB"),
        help="Real Drive Sim RGB modality root (defaults to $REAL_DRIVE_SIM_RGB).",
    )
    parser.add_argument(
        "--count", type=int, default=3, help="Number of frames to show."
    )
    args = parser.parse_args()

    if not args.path:
        parser.error("set REAL_DRIVE_SIM_RGB or pass --path /path/to/rgb")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("This example needs matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    dataset = MultiModalDataset(
        modalities={"rgb": Modality(args.path, loader=real_drive_sim.rgb)},
    )
    print(f"Dataset size: {len(dataset)} frames")

    count = min(args.count, len(dataset))
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)

    for i, ax in enumerate(axes[0]):
        sample = dataset[i]
        # (3, H, W) float32 [0, 1] -> (H, W, 3) for imshow
        ax.imshow(sample["rgb"].permute(1, 2, 0).numpy())
        ax.set_title(sample["id"])
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
