"""Pre-defined loader functions for common datasets.

Each dataset has its own submodule with plain functions following the
``Callable[[str], Any]`` signature expected by :class:`~euler_loading.Modality`.

Usage::

    from euler_loading.loaders import vkitti2
    from euler_loading import Modality, MultiModalDataset

    dataset = MultiModalDataset(
        modalities={
            "rgb":   Modality("/data/vkitti2/rgb",   loader=vkitti2.rgb),
            "depth": Modality("/data/vkitti2/depth", loader=vkitti2.depth),
        },
        read_intrinsics=vkitti2.read_intrinsics,
        read_extrinsics=vkitti2.read_extrinsics,
    )

Available submodules:

- :mod:`euler_loading.loaders.vkitti2` — Virtual KITTI 2
"""
