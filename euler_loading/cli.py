"""Command line entry point: dry-run the loaders a dataset declares.

::

    euler-loading /data/vkitti2
    python -m euler_loading /data/vkitti2

Both forms take one path — a modality root, a ``.zip`` archive, or a folder
holding several of them — resolve the loader each ds-crawler artifact set
below it declares, decode a few files with it, and print what came back.
Nothing is written.  See :mod:`euler_loading.dry_run` for the API behind it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from .dry_run import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_SAMPLES,
    DryRunReport,
    ModalityCheck,
    dry_run,
)

__all__ = ["main"]

_FIELD_WIDTH = 9
_INDENT = "  "
_LOGGED_PACKAGES = ("euler_loading", "ds_crawler")


class _StderrHandler(logging.Handler):
    """Write to whichever stream is ``sys.stderr`` at the time of the record.

    A handler that captures the stream up front outlives a redirected or
    closed one, which matters because ``main`` is importable and can be
    called more than once in a process.
    """

    def emit(self, record: logging.LogRecord) -> None:
        stream = sys.stderr
        if stream is None or getattr(stream, "closed", False):
            return
        try:
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:  # pragma: no cover - defensive, as logging requires
            self.handleError(record)


def setup_logging(verbose: bool) -> None:
    """Route euler-loading and ds-crawler logs to stderr, unbuffered.

    Only those two loggers are touched — never the root logger — so calling
    :func:`main` in-process leaves a host application's logging alone.  Levels
    are lowered only for ``--verbose``; otherwise the inherited level stands
    and this just gives warnings a visible prefix.
    """
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    for name in _LOGGED_PACKAGES:
        logger = logging.getLogger(name)
        for existing in list(logger.handlers):
            if isinstance(existing, _StderrHandler):
                logger.removeHandler(existing)
        logger.addHandler(handler)
        if verbose:
            logger.setLevel(logging.DEBUG)


def _field(label: str, value: str) -> str:
    return f"{_INDENT}{label.ljust(_FIELD_WIDTH)} {value}"


def _format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    return f"{seconds * 1000:.1f} ms"


def _contract_line(check: ModalityCheck) -> Optional[str]:
    parts = []
    if check.dataset_id:
        parts.append(check.dataset_id)
    if check.dataset_name and check.dataset_name != check.dataset_id:
        parts.append(f'"{check.dataset_name}"')
    if check.modality_key:
        parts.append(f"modality key: {check.modality_key}")
    return ", ".join(parts) if parts else None


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _index_line(check: ModalityCheck) -> Optional[str]:
    parts = []
    if check.file_count is not None:
        parts.append(_plural(check.file_count, "file", "files"))
    if check.split:
        parts.append(f"split: {check.split}")
    if check.splits:
        parts.append(f"available splits: {', '.join(check.splits)}")
    return "; ".join(parts) if parts else None


def _render_modality(check: ModalityCheck) -> list[str]:
    status = "ok" if check.ok else "FAILED"
    lines = [f"{check.label}  [{status}]"]
    lines.append(_field("path", check.path))

    contract = _contract_line(check)
    if contract:
        lines.append(_field("contract", contract))

    if check.loader:
        loader = check.loader
        if check.declared_loader:
            loader = f"{check.declared_loader} -> {loader}"
        if check.hierarchical:
            loader = f"{loader} (hierarchical)"
        lines.append(_field("loader", loader))
    elif check.declared_loader:
        lines.append(_field("loader", f"{check.declared_loader} (unresolved)"))

    if check.writer:
        lines.append(_field("writer", check.writer))

    index = _index_line(check)
    if index:
        lines.append(_field("index", index))

    if check.samples:
        lines.append(
            _field("decoded", f"{check.decoded}/{len(check.samples)}")
        )
        for sample in check.samples:
            name = sample.path or sample.full_id
            if sample.ok:
                detail = f"{sample.value}  ({_format_duration(sample.seconds)})"
            else:
                detail = f"FAILED  {sample.error}"
            lines.append(f"{_INDENT * 2}{name}  ->  {detail}")

    for warning in check.warnings:
        lines.append(_field("warning", warning))
    for error in check.errors:
        lines.append(_field("error", error))
    return lines


def render(report: DryRunReport) -> str:
    """Render a dry-run report as plain text."""
    variant_label = "torch tensors" if report.variant == "gpu" else "numpy arrays"
    lines = [
        f"euler-loading dry-run: {report.root}",
        f"loader variant: {report.variant} ({variant_label})",
    ]

    for check in report.modalities:
        lines.append("")
        lines.extend(_render_modality(check))

    if report.modalities:
        attempted = sum(len(check.samples) for check in report.modalities)
        decoded = sum(check.decoded for check in report.modalities)
        passed = sum(1 for check in report.modalities if check.ok)
        checked = _plural(len(report.modalities), "modality", "modalities")
        summary = (
            f"{checked} checked: "
            f"{passed} ok, {len(report.modalities) - passed} failed"
        )
        if attempted:
            summary = f"{summary}; {decoded}/{attempted} files decoded"
        if report.common_ids is not None:
            shared = _plural(report.common_ids, "shared file ID", "shared file IDs")
            summary = f"{summary}; {shared}"
        lines.extend(["", summary])

    if report.warnings or report.errors:
        lines.append("")
        lines.extend(f"warning: {warning}" for warning in report.warnings)
        lines.extend(f"error: {error}" for error in report.errors)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler-loading",
        description=(
            "Dry-run the loaders a dataset declares. Resolves the loader "
            "named by each modality's ds-crawler dataset-head contract "
            "(addons.euler_loading), decodes a few indexed files with it and "
            "reports what came back. Writes nothing."
        ),
        epilog=(
            "Exits 0 when every modality resolved a loader and decoded every "
            "file it was asked to, and 1 otherwise."
        ),
    )
    parser.add_argument(
        "path",
        help=(
            "Modality root (directory or .zip), or a folder holding several "
            "of them. Accepts the inline ':split' and '#scope=NAME' "
            "selectors that Modality accepts."
        ),
    )
    parser.add_argument(
        "--split",
        metavar="NAME",
        default=None,
        help="Dry-run this ds-crawler split instead of the canonical index.",
    )
    parser.add_argument(
        "--scope",
        dest="metadata_scope",
        metavar="SCOPE",
        default=None,
        help=(
            "Read .ds_crawler/SCOPE/ only, instead of every artifact set each "
            "root carries."
        ),
    )
    samples = parser.add_mutually_exclusive_group()
    samples.add_argument(
        "-n",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        metavar="N",
        help=(
            "Files to decode per modality, spread evenly across the index "
            f"(default: {DEFAULT_SAMPLES}). 0 resolves loaders without "
            "decoding anything."
        ),
    )
    samples.add_argument(
        "--all",
        action="store_true",
        help="Decode every indexed file. Slow, but checks the whole archive.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Use the NumPy (CPU) loaders instead of the torch (GPU) ones "
            "automatic resolution picks."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        metavar="N",
        help=(
            "Directory levels below PATH to search for dataset roots "
            f"(default: {DEFAULT_MAX_DEPTH}). Roots are never descended into."
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print the report as JSON instead of text.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug output, including tracebacks for failed loads.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the dry-run command. Returns the process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # ``dry-run`` is the only command; accept it spelled out so the invocation
    # still reads correctly if others are added later.
    if argv and argv[0] == "dry-run":
        argv = argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.samples < 0:
        parser.error("--samples must be 0 or greater; use --all for every file")
    if args.max_depth < 0:
        parser.error("--max-depth must be 0 or greater")

    setup_logging(args.verbose)

    try:
        report = dry_run(
            args.path,
            split=args.split,
            metadata_scope=args.metadata_scope,
            samples=None if args.all else args.samples,
            variant="cpu" if args.cpu else "gpu",
            max_depth=args.max_depth,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    sys.exit(main())
