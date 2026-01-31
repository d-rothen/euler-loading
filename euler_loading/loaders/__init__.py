"""Pre-defined loader functions for common datasets.

Each dataset has its own submodule with plain functions following the
``Callable[[str], Any]`` signature expected by :class:`~euler_loading.Modality`.

Loaders come in two variants:

- **gpu** – return ``torch.Tensor`` in CHW layout, ready for GPU training.
- **cpu** – return ``numpy.ndarray`` in HWC layout, for CPU-based processing.

Usage::

    # GPU loaders (torch tensors)
    from euler_loading.loaders.gpu import vkitti2

    # CPU loaders (numpy arrays)
    from euler_loading.loaders.cpu import vkitti2

    # Default (GPU) – backward-compatible shorthand
    from euler_loading.loaders import vkitti2

Available submodules:

- :mod:`euler_loading.loaders.gpu.vkitti2` — Virtual KITTI 2 (torch)
- :mod:`euler_loading.loaders.cpu.vkitti2` — Virtual KITTI 2 (numpy)
"""
