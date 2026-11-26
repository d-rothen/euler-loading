# Contributing

Thanks for considering a contribution. Bug reports, loaders for new datasets and
documentation fixes are all welcome.

## Development setup

```bash
git clone https://github.com/d-rothen/euler-loading.git
cd euler-loading
pip install -e ".[gpu,dev]"
```

The `[gpu]` extra installs PyTorch, which the GPU loaders and most of the test
suite need. `[dev]` installs pytest.

## Running tests

```bash
pytest                  # unit tests — mocked, no data needed
pytest -m real          # integration tests against real on-disk datasets
```

`-m 'not real'` is the configured default, so the real-dataset tests are
deselected unless you ask for them. They also need the relevant environment
variable set, and skip themselves when the paths are absent:

```bash
export VKITTI2_ROOT=/path/to/vkitti2
pytest -m real
```

Please keep the unit suite runnable without any dataset on disk.

## Adding a loader

A loader for your own dataset does not need to live here: register it from
your own package (README, "Extending: your own loaders"). Built-in loaders
use exactly the same mechanism.

Loaders live in `euler_loading/loaders/{cpu,gpu}/<dataset>.py`. Each dataset
module should exist in both variants: `gpu` returns `torch.Tensor` in CHW
layout, `cpu` returns `numpy.ndarray` in HWC.

1. **Write the function.** The signature is
   `(path: str | BinaryIO, meta: dict | None = None, *, attributes: dict | None = None)`.
   Accept `BinaryIO` as well as `str` so zip-backed modalities work — the shared
   helpers in `loaders/_writer_utils.py` handle both.
2. **Annotate it** with `@modality_meta(...)` from `euler_loading.annotations`,
   declaring type, dtype, shape, unit, output range and accepted file formats.
   This annotation is what makes the loader discoverable and self-describing:
   the capability report and the receipts read it.
3. **Add a writer** where a round trip makes sense, named `write_<function>`, so
   writer resolution finds it automatically.
4. **Register it.** Add `<dataset> = _variants("<dataset>")` to
   `euler_loading/loaders/builtin.py` and the matching line under
   `[project.entry-points."euler.loaders"]` in `pyproject.toml`, then reinstall
   (`pip install -e .`) so the entry point exists in your environment.
   `tests/test_registry.py` checks that the two lists agree and that every
   registered module imports.
5. **Add tests** in `tests/test_loaders.py`, covering both variants.
6. **Check it against real data** with the dry run, which resolves the loader
   the dataset's `dataset-head.json` declares and decodes files with it:

   ```bash
   euler-loading /path/to/dataset --all
   ```

7. **Update [`docs/loaders.md`](docs/loaders.md)** with the new functions.

Loader names resolve only through installed entry points, so run the tests
from an installed checkout (`pip install -e ".[gpu,dev]"`), not with
`PYTHONPATH` pointing at a source tree.

## Style

- No formatter or linter is enforced in CI; match the surrounding code.
- Type annotations on public functions, `from __future__ import annotations` at
  the top of each module.
- Keep the package importable without torch. Guard torch imports the way
  `dataset.py` and the `cpu` loaders do — the CPU path must work on a
  torch-free install.
- Python 3.9 is the floor. CI runs the suite on 3.9 through 3.13, and
  `tests/test_python_compat.py` guards against syntax that would break the
  older versions.

## Releasing

Releases are published to PyPI by
[`.github/workflows/publish.yml`](.github/workflows/publish.yml) via trusted
publishing, triggered by a `v*` tag:

```bash
# bump `version` in pyproject.toml first, and commit it
git tag v2.22.0
git push origin v2.22.0
```

The tag and the version in `pyproject.toml` must match.
