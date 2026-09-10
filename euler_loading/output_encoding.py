"""Explicit versioned output codecs shared by writer metadata and reload."""

from __future__ import annotations

import io
from copy import deepcopy

import numpy as np
from euler_dataset_contract import OutputEncoding
from PIL import Image
from PIL import __version__ as pillow_version


def output_encoding(kind="npy", *, scale=None):
    if kind == "npy":
        scale, limits = 1.0, [-9007199254740991, 9007199254740991]
    elif kind == "png_depth16":
        scale = 0.001 if scale is None else scale
        limits = [0, 65535 * scale]
    elif kind == "png_rgb8":
        scale, limits = 1 / 255, [0, 1]
    elif kind == "png_mask8":
        scale, limits = 1, [0, 1]
    else:
        raise ValueError(f"unsupported output encoding {kind!r}")
    versions = {"numpy": np.__version__}
    if kind != "npy":
        versions["pillow"] = pillow_version
    return OutputEncoding(
        {
            "id": kind,
            "version": "1.0",
            "scale": scale,
            "range": limits,
            "clipping": "reject",
            "rounding": "ties_to_even",
            "versions": versions,
        }
    )


def check_encoding(encoding):
    encoding = OutputEncoding(encoding)
    expected = output_encoding(encoding["id"], scale=encoding["scale"])
    if expected != encoding:
        raise ValueError("unsupported output encoding parameters/backend versions")
    return encoding


def encoding_profiles(profile, encoding):
    encoding = check_encoding(encoding)
    profile = deepcopy(dict(profile))
    kind = encoding["id"]
    decoded, storage = deepcopy(profile), deepcopy(profile)
    if kind == "npy":
        return {"decoded": decoded, "storage": storage}
    supported = {
        "png_rgb8": ("image", "HWC", "float32"),
        "png_depth16": ("depth", "HW", "float32"),
        "png_mask8": ("mask", "HW", "bool"),
    }
    if (profile["kind"], profile["layout"], profile["dtype"]) != supported[kind]:
        raise ValueError(f"{kind} does not support this output profile; use npy")
    if kind == "png_rgb8" and profile["shape"][-1] != 3:
        raise ValueError("RGB8 requires exactly three channels")
    storage["dtype"] = "uint16" if kind == "png_depth16" else "uint8"
    storage["unit"] = "code_value"
    storage["invalid"]["non_finite"] = False
    sentinel = storage["invalid"]["sentinel"]
    if sentinel is not None:
        if not encoding["range"][0] <= sentinel <= encoding["range"][1]:
            raise ValueError(
                "finite invalid sentinel is outside the output encoding range"
            )
        storage["invalid"]["sentinel"] = int(np.rint(sentinel / encoding["scale"]))
        decoded_sentinel = np.array(
            storage["invalid"]["sentinel"] * encoding["scale"], dtype=profile["dtype"]
        ).item()
        if decoded_sentinel != np.array(sentinel, dtype=profile["dtype"]).item():
            raise ValueError(
                "output encoding cannot preserve the declared invalid sentinel; use npy"
            )
    return {"decoded": decoded, "storage": storage}


def encode_output(value, encoding, profile):
    encoding = check_encoding(encoding)
    profiles = encoding_profiles(profile, encoding)
    array = (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )
    buffer = io.BytesIO()
    if encoding["id"] == "npy":
        # Memory strides are not part of the decoded profile. Give equal arrays
        # one storage order so idempotent writes and publication replay agree.
        np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    else:
        low, high = encoding["range"]
        if not np.isfinite(array).all() or np.any(array < low) or np.any(array > high):
            raise ValueError(
                "output values exceed declared encoding range; clipping is forbidden"
            )
        stored = np.rint(array / encoding["scale"]).astype(profiles["storage"]["dtype"])
        Image.fromarray(stored).save(buffer, format="PNG")
    raw = buffer.getvalue()
    decoded = decode_output(io.BytesIO(raw), encoding, profiles["decoded"])
    sentinel = profile["invalid"]["sentinel"]
    if (
        encoding["id"] != "npy"
        and sentinel is not None
        and not np.array_equal(array == sentinel, decoded == sentinel)
    ):
        raise ValueError(
            "output quantization would change invalid-pixel classification; use npy"
        )
    tolerance = encoding["scale"] / 2 + np.finfo(np.float32).eps * max(
        abs(x) for x in encoding["range"]
    )
    if encoding["id"] != "npy" and not np.allclose(
        decoded, array, atol=tolerance, rtol=0
    ):
        raise ValueError("output codec exceeds declared quantization tolerance")
    return raw, decoded, profiles


def decode_output(source, encoding, profile):
    encoding = check_encoding(encoding)
    if encoding["id"] == "npy":
        result = np.load(source, allow_pickle=False)
    else:
        with Image.open(source) as image:
            stored = np.array(image)
        result = (stored.astype(np.float64) * encoding["scale"]).astype(
            profile["dtype"]
        )
    if list(result.shape) != profile["shape"] or result.dtype.name != profile["dtype"]:
        raise ValueError("stored artifact disagrees with decoded output profile")
    return result
