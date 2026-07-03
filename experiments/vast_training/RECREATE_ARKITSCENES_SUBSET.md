# Recreate Synthetic ARKitScenes Subset

This note exists for one very specific recovery path:

- the Vast machine only has raw ARKitScenes `Training` scenes
- the official raw `Validation` split is missing
- we still want to recreate the same synthetic subset experiment used on 2026-05-26

The recovery helper is:

- `recreate_arkitscenes_subset_split.py`

It bakes in the same 20 held-out scene IDs that were used for the synthetic `Validation` split, rebuilds:

- `/workspace/datasets/arkitscenes_raw_trainval_subset`

and can also rebuild:

- `/workspace/datasets/arkitscenes_processed`

## What this is

This is **not** the official ARKitScenes train/validation protocol.

It is a synthetic holdout made from raw `Training` scenes only. That is fine for reproducing the subset experiment and getting back to a known machine state, but it should be described honestly in the thesis.

## Typical usage on a fresh Vast machine

```bash
export PYTHON_BIN=/venv/main/bin/python
export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

$PYTHON_BIN /workspace/vast_training/recreate_arkitscenes_subset_split.py --force
```

That command:

1. recreates `/workspace/datasets/arkitscenes_raw_trainval_subset`
2. rebuilds `/workspace/datasets/arkitscenes_processed`
3. prints metadata counts for `Training` and `Test`

## Assumptions

This helper expects the raw Training download to exist at:

```text
/workspace/datasets/arkitscenes_raw/raw/Training
```

If your raw files live somewhere else, override it:

```bash
$PYTHON_BIN /workspace/vast_training/recreate_arkitscenes_subset_split.py \
  --raw-training-root /some/other/path/Training \
  --force
```

## Safe mode

If the target subset or processed directories already exist, the script refuses to overwrite them unless `--force` is passed.

That is intentional. The main job of this helper is to make recovery reproducible without accidentally wiping a still-good dataset.
