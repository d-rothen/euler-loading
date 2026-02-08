"""Shared fixtures for euler-loading tests."""

from __future__ import annotations

from typing import Any

import pytest


def _make_file(file_id: str, path: str) -> dict[str, Any]:
    """Build a minimal ds-crawler file entry."""
    return {
        "id": file_id,
        "path": path,
        "path_properties": {},
        "basename_properties": {},
    }


def _nest(keys: list[str], leaf: dict[str, Any]) -> dict[str, Any]:
    """Build a nested ``{"children": {...}}`` structure from *keys* down to *leaf*."""
    node = leaf
    for key in reversed(keys):
        node = {"children": {key: node}}
    return node


@pytest.fixture()
def flat_index() -> dict[str, Any]:
    """Index with files at the root node (no hierarchy)."""
    return {
        "dataset": {
            "files": [
                _make_file("frame-001", "001.png"),
                _make_file("frame-002", "002.png"),
            ]
        }
    }


@pytest.fixture()
def hierarchical_index() -> dict[str, Any]:
    """Index with calibration at the camera level and files at the frame level.

    Structure::

        dataset
        └── scene:Scene01
            └── camera:Camera_0    (camera_intrinsics, camera_extrinsics)
                ├── frame:00001    (file)
                └── frame:00002    (file)
    """
    return {
        "dataset": {
            "children": {
                "scene:Scene01": {
                    "children": {
                        "camera:Camera_0": {
                            "camera_intrinsics": "Scene01/Camera_0_intr.txt",
                            "camera_extrinsics": "Scene01/Camera_0_extr.txt",
                            "children": {
                                "frame:00001": {
                                    "files": [
                                        _make_file(
                                            "scene-Scene01+camera-Camera_0+frame-00001",
                                            "Scene01/Camera_0/00001.png",
                                        )
                                    ]
                                },
                                "frame:00002": {
                                    "files": [
                                        _make_file(
                                            "scene-Scene01+camera-Camera_0+frame-00002",
                                            "Scene01/Camera_0/00002.png",
                                        )
                                    ]
                                },
                            },
                        }
                    }
                }
            }
        }
    }


@pytest.fixture()
def multi_camera_index() -> dict[str, Any]:
    """Index with two cameras — Camera_0 has calibration, Camera_1 does not."""
    return {
        "dataset": {
            "children": {
                "scene:Scene01": {
                    "children": {
                        "camera:Camera_0": {
                            "camera_intrinsics": "Scene01/Camera_0_intr.txt",
                            "children": {
                                "frame:00001": {
                                    "files": [
                                        _make_file(
                                            "scene-Scene01+camera-Camera_0+frame-00001",
                                            "Scene01/Camera_0/00001.png",
                                        )
                                    ]
                                },
                            },
                        },
                        "camera:Camera_1": {
                            "children": {
                                "frame:00001": {
                                    "files": [
                                        _make_file(
                                            "scene-Scene01+camera-Camera_1+frame-00001",
                                            "Scene01/Camera_1/00001.png",
                                        )
                                    ]
                                },
                            },
                        },
                    },
                }
            }
        }
    }


@pytest.fixture()
def override_calibration_index() -> dict[str, Any]:
    """Index where a child node overrides the parent's calibration.

    Scene level has ``scene_intr.txt``, but Camera_0 overrides with its own.
    """
    return {
        "dataset": {
            "children": {
                "scene:Scene01": {
                    "camera_intrinsics": "Scene01/scene_intr.txt",
                    "children": {
                        "camera:Camera_0": {
                            "camera_intrinsics": "Scene01/Camera_0_intr.txt",
                            "children": {
                                "frame:00001": {
                                    "files": [
                                        _make_file(
                                            "s01-c0-f001",
                                            "Scene01/Camera_0/00001.png",
                                        )
                                    ]
                                },
                            },
                        },
                        "camera:Camera_1": {
                            "children": {
                                "frame:00001": {
                                    "files": [
                                        _make_file(
                                            "s01-c1-f001",
                                            "Scene01/Camera_1/00001.png",
                                        )
                                    ]
                                },
                            },
                        },
                    },
                }
            }
        }
    }


@pytest.fixture()
def hierarchical_modality_index() -> dict[str, Any]:
    """Index for a hierarchical modality (e.g. intrinsics) with files at the
    variation level, one level above where regular-modality files sit.

    Structure::

        dataset
        └── Scene01
            └── sunset
                └── file: intrinsic   (intrinsic.txt)
    """
    return {
        "dataset": {
            "children": {
                "Scene01": {
                    "children": {
                        "sunset": {
                            "files": [
                                _make_file(
                                    "intrinsic",
                                    "Scene01/sunset/intrinsic.txt",
                                )
                            ]
                        }
                    }
                }
            }
        }
    }


@pytest.fixture()
def deep_regular_index() -> dict[str, Any]:
    """Regular-modality index with files one level deeper than the
    hierarchical modality fixture (camera level under variation).

    Structure::

        dataset
        └── Scene01
            └── sunset
                └── Camera_0
                    ├── file: frame-001
                    └── file: frame-002
    """
    return {
        "dataset": {
            "children": {
                "Scene01": {
                    "children": {
                        "sunset": {
                            "children": {
                                "Camera_0": {
                                    "files": [
                                        _make_file(
                                            "frame-001",
                                            "Scene01/sunset/Camera_0/001.png",
                                        ),
                                        _make_file(
                                            "frame-002",
                                            "Scene01/sunset/Camera_0/002.png",
                                        ),
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    }


@pytest.fixture()
def multi_level_hierarchical_index() -> dict[str, Any]:
    """Hierarchical index with files at two levels — scene and variation.

    Structure::

        dataset
        └── Scene01           (file: scene_meta)
            └── sunset        (file: intrinsic)
    """
    return {
        "dataset": {
            "children": {
                "Scene01": {
                    "files": [
                        _make_file("scene_meta", "Scene01/meta.json"),
                    ],
                    "children": {
                        "sunset": {
                            "files": [
                                _make_file(
                                    "intrinsic",
                                    "Scene01/sunset/intrinsic.txt",
                                )
                            ]
                        }
                    },
                }
            }
        }
    }


def dummy_loader(path: str, meta: dict[str, Any] | None = None) -> str:
    """Trivial loader that returns the path — avoids needing real files."""
    return f"loaded:{path}"
