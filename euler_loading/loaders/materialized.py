"""Loads stored bytes once. Transform receipts are history, never instructions."""

from __future__ import annotations

import io
from pathlib import Path

from euler_loading.output_encoding import decode_output
from euler_loading.receipts import _digest_bytes, array_digest

from ._annotations import modality_meta


@modality_meta(
    modality_type="generic",
    dtype="declared",
    shape="declared",
    file_formats=[".npy", ".png"],
)
def data(source, meta=None, *, attributes=None):
    if not attributes or "output_encoding" not in attributes:
        raise ValueError(
            "materialized decoder requires per-file output encoding metadata"
        )
    info = attributes["output_encoding"]
    raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    if _digest_bytes(raw) != info["artifact_digest"]:
        raise ValueError("stored artifact bytes changed")
    value = decode_output(io.BytesIO(raw), info["encoding"], info["decoded"])
    if array_digest(value) != info["decoded_digest"]:
        raise ValueError("stored decoded artifact digest mismatch")
    return value
