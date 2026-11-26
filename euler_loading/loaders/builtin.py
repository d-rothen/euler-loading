"""euler-loading's own ``euler.loaders`` entry-point values.

``pyproject.toml`` registers each of these under ``[project.entry-points.
"euler.loaders"]``, exactly as a third-party package registers its loaders;
resolution never reads this module directly, only through the registry
(:mod:`euler_loading.registry`). Importing it is cheap: the values only name
modules, which are imported when a dataset head asks for them.

Adding a dataset: write ``loaders/gpu/<name>.py`` and ``loaders/cpu/<name>.py``,
add ``<name> = _variants("<name>")`` here and the matching entry point in
``pyproject.toml``, then reinstall (``pip install -e .``) so the entry point
exists. ``tests/test_registry.py`` checks the two lists agree.
"""

from __future__ import annotations

API_VERSION = "1.0"


def _variants(name: str) -> dict[str, str]:
    return {
        "api_version": API_VERSION,
        "gpu": f"euler_loading.loaders.gpu.{name}",
        "cpu": f"euler_loading.loaders.cpu.{name}",
    }


acdc = _variants("acdc")
cerema = _variants("cerema")
driving_stereo = _variants("driving_stereo")
generic = _variants("generic")
generic_dense_depth = _variants("generic_dense_depth")
hypersim = _variants("hypersim")
kitti_carla = _variants("kitti_carla")
muses = _variants("muses")
nuscenes = _variants("nuscenes")
princeton_dense = _variants("princeton_dense")
real_drive_sim = _variants("real_drive_sim")
scatternerf = _variants("scatternerf")
shift = _variants("shift")
synscapes = _variants("synscapes")
urbansyn = _variants("urbansyn")
vkitti2 = _variants("vkitti2")

#: Stored outputs decode to whatever their per-file encoding declares, so one
#: module serves both variants.
materialized = {
    "api_version": API_VERSION,
    "gpu": "euler_loading.loaders.materialized",
    "cpu": "euler_loading.loaders.materialized",
}
