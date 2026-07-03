# Dataset Shortlist For Fast3R Fine-Tuning

For the current thesis sprint, **ScanNet is the primary training dataset**.

Everything else in this file should be treated as optional comparison material, not the main path.

This list is biased toward what already has loader support in the local Fast3R repo and what is likely to help a phone-video indoor reconstruction workflow.

## Primary dataset for the current run

### 1. ScanNet

- Loader support:
  `Scannet` from `fast3r.data.components.spann3r_datasets.scannet`
- Good fit for:
  indoor multiview reconstruction, walls, planes, corners, room-scale scenes
- Why it is attractive:
  it matches the domain we actually care about right now
- Current status in this bundle:
  the Vast training scripts and configs are centered on ScanNet

## Optional secondary datasets

### 2. ARKitScenes

- Loader support: `ARKitScenes`, `ARKitScenes_Multiview`
- Good fit for: indoor rooms, handheld device motion, RGB-D aligned geometry
- Why it is attractive:
  gives you indoor scene structure closer to room scans and phone capture
- Practical status in this bundle:
  `vast_run_training.ksh` can optionally download the official raw data on Vast
- Caveat:
  preprocessing/storage can be heavy, and the Fast3R loaders expect `arkitscenes_processed` rather than just raw download

## 3. ScanNet++

- Loader support: `ScanNetpp`, `ScanNetpp_Multiview`
- Good fit for: high-quality indoor multiview geometry
- Why it is attractive:
  stronger scene consistency and cleaner indoor geometry than many casual datasets
- Caveat:
  access terms and preprocessing effort

## 4. MegaDepth

- Loader support: `MegaDepth`, `MegaDepth_Multiview`
- Good fit for: general geometric robustness and harder viewpoint changes
- Why it is attractive:
  useful for keeping the model from over-specializing to only neat indoor sequences
- Caveat:
  weaker match to room-scale handheld videos than ARKitScenes/ScanNet++

## 5. Co3D

- Loader support: `Co3d`, `Co3d_Multiview`
- Good fit for: object-centric multiview learning
- Why it is attractive:
  helps if you later care about cleaner object geometry or turntable-like captures
- Caveat:
  less aligned to full-room reconstruction

## 6. BlendedMVS

- Loader support: `BlendedMVS`, `BlendedMVS_Multiview`
- Good fit for: classical MVS-style geometry supervision
- Why it is attractive:
  already used in the official-style training mixes
- Caveat:
  more useful as a support dataset than as the first domain dataset

## Datasets already used mainly for validation/evaluation in the repo

### DTU

- Useful for:
  geometric validation and controlled benchmarking
- Not my first pick for training your target domain

### SevenScenes

- Useful for:
  indoor pose/reconstruction evaluation
- Better as evaluation than primary training

### Neural RGB-D

- Useful for:
  evaluation on indoor RGB-D reconstruction
- Better as evaluation than first training source

## Other supported datasets in the repo

### WildRGBD

- Potentially interesting for:
  more unconstrained RGB-D captures
- Worth investigating if you want less polished real-world data

### Habitat

- Synthetic
- Useful if you need lots of cheap extra multiview data
- Risk:
  domain gap

### StaticThings3D / Waymo

- Supported in the repo
- Not my first recommendation for your indoor handheld objective

## Best first training mix for your use case

If the target is room/interior handheld reconstruction and the deadline is short, I would start with:

1. `ScanNet`
2. optionally add `ARKitScenes` later
3. optionally add `ScanNet++` later

That gives you:

- indoor geometry
- real capture noise
- some extra viewpoint diversity

## Best first evaluation set

After head-only fine-tuning, I’d compare on:

1. `SevenScenes`
2. `NRGBD`
3. `DTU`

and then of course on your own videos.

## What you should decide before downloading

- indoor-only vs general-purpose
- object-centric vs whole-room
- allowed licenses for your thesis/demo usage
- how much preprocessing/storage you can tolerate

## Practical recommendation

If you want the lowest-risk first paid run:

- train on `ScanNet`
- validate first on `ScanNet val`
- use `SevenScenes` and `NRGBD` only as secondary indoor checks
