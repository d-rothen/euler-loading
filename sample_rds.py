"""Sample script: load and display RGB frames from a Real Drive Sim dataset."""

import matplotlib.pyplot as plt

from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.gpu import real_drive_sim

RGB_PATH = "/Volumes/Volume/Datasets/real-drive-sim/rgb"

dataset = MultiModalDataset(
    modalities={
        "rgb": Modality(RGB_PATH, loader=real_drive_sim.rgb),
    },
)

print(f"Dataset size: {len(dataset)} frames")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    sample = dataset[i]
    # (3, H, W) float32 [0,1] -> (H, W, 3) for imshow
    ax.imshow(sample["rgb"].permute(1, 2, 0).numpy())
    ax.set_title(sample["id"])
    ax.axis("off")

fig.tight_layout()
plt.show()


