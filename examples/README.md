# Examples

Runnable scripts against real datasets. Each one takes its dataset root from an
environment variable or a CLI flag — nothing is hardcoded, so point them at
wherever your copy of the data lives.

Every modality root must already be indexed by
[ds-crawler](https://github.com/d-rothen/ds-crawler); euler-loading reads the
`.ds_crawler/` artifacts it produces.

| Script | What it shows | Configure with |
|---|---|---|
| [`vkitti2_sample.py`](vkitti2_sample.py) | Intersecting four VKITTI 2 modalities, and the CPU vs GPU loader variants | `VKITTI2_ROOT` or `--root` |
| [`real_drive_sim_preview.py`](real_drive_sim_preview.py) | Rendering loaded RGB frames with matplotlib | `REAL_DRIVE_SIM_RGB` or `--path` |

```bash
export VKITTI2_ROOT=/path/to/vkitti2

python examples/vkitti2_sample.py          # torch tensors (CHW)
python examples/vkitti2_sample.py --cpu    # numpy arrays  (HWC)
```

`real_drive_sim_preview.py` additionally needs `matplotlib`, which is not a
dependency of euler-loading:

```bash
pip install matplotlib
```
