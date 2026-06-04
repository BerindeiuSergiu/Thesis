# Thesis Knowledge Base and Structured Marker Document

## Source Audit Status

- Primary structure source used: [template](/D:/GitRepos/Thesis/thesis_docs/document/template)
- Requested style source missing from repo at time of audit: `thesis_docs/document/my_work.docx`
- Consequence: chapter planning below is complete, but author-style calibration is still blocked and must be revisited once `my_work.docx` is available
- Fallback style evidence currently available in repo: engineering notes and implementation-facing prose in [src/README.md](/D:/GitRepos/Thesis/src/README.md), [src/EXPERIMENTS_AUDIT.md](/D:/GitRepos/Thesis/src/EXPERIMENTS_AUDIT.md), [experiments/geometry_test/IMPLEMENTATION_NOTES.txt](/D:/GitRepos/Thesis/experiments/geometry_test/IMPLEMENTATION_NOTES.txt), [experiments/vast_training/README.md](/D:/GitRepos/Thesis/experiments/vast_training/README.md)
- Important thesis honesty marker: GA/global-head fine-tuning now has local evidence for one completed shortened ARKitScenes run (`arkitscenes_10hour_ga_head`), including metrics CSV and checkpoint metadata. However, a separate pretrained-versus-fine-tuned paired comparison bundle was not found locally, so thesis wording must distinguish "validation loss decreased during the fine-tuning run" from "controlled pretrained-versus-fine-tuned benchmark improvement."

# Project Title

- Candidate title markers:
  - Fast3R-Based Indoor 3D Reconstruction Pipeline for Video-to-Scene Generation
  - Geometry-Aware Fast3R Reconstruction Pipeline with Scaled Gaussian and Mesh Outputs
  - Practical Indoor 3D Reconstruction from Monocular Video Using Fast3R, Scale Normalization, and Dual Outputs
- Core system identity:
  - desktop wrapper around Fast3R inference
  - video input
  - geometry-aware frame selection
  - scale normalization
  - dual-output reconstruction: Gaussian splat + mesh
  - performance focus: reduce time-to-inspectable-scene
  - scene metadata and reusable output packaging
- Evidence:
  - [src/README.md](/D:/GitRepos/Thesis/src/README.md)
  - [src/pipeline/fast3r_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/fast3r_pipeline.py)
  - [experiments/beta_pipeline_testing/README.md](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/README.md)
- Missing:
  - final thesis title decision
  - explicit advisor-approved wording

# 1. Introduction

## Problem and motivation

- Problem marker: converting a handheld room video into a usable 3D scene is difficult because raw frame streams contain blur, redundancy, weak texture areas, inconsistent geometry, and no guaranteed real-world scale
- Problem marker: dense reconstruction alone is not enough; downstream usage needs structure, scale, and output forms that are actually reusable in viewers, simulation, or engines
- Motivation marker: monocular/depth-per-frame approaches were fast but geometrically weak; sparse SfM was geometrically cleaner but too sparse for the target workflow
- Performance marker: reducing the time-to-inspectable-scene is a central application objective; conventional iterative reconstruction and optimization-heavy Gaussian-splat workflows can require substantial waiting time before the result becomes useful to inspect
- Comparative evidence:
  - [experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md) compares earlier COLMAP SfM and MiDaS-style depth pipeline behavior
  - COLMAP result marker: clean sparse geometry, 1,230 points, 24 reconstructed images from 100 frames, reported runtime approximately 8 minutes
  - MiDaS result marker: dense but noisy geometry, 664k+ points, poor mesh topology, ambiguous scale
- Main project motivation evolved toward:
  - keeping the multiview geometric consistency benefits of a learned reconstruction model
  - reducing poor input frames before inference
  - recovering usable real-world scale after inference
  - producing both photorealistic and geometric outputs from the same reconstruction
  - producing an inspectable scene quickly enough to support a practical local feedback loop
- Thesis contribution marker:
  - not just "use Fast3R"
  - adapt Fast3R into an engineering pipeline with geometry-aware selection, confidence-based retention, scale normalization, and output branching
- Good figures:
  - [fast3r_geometry_aware_pipeline.svg](/D:/GitRepos/Thesis/thesis_docs/writing/fast3r_geometry_aware_pipeline.svg)
  - screenshots/renders from `scene_*` outputs once captured for thesis figures
- Missing:
  - direct problem statement draft in author voice
  - one or two user-story examples of why both Gaussian and mesh outputs matter

## Project objectives

- Objective marker: build a practical indoor reconstruction pipeline around Fast3R for room-scale videos
- Objective marker: reduce time-to-inspectable-scene through multi-view inference, representative-frame selection, and shared downstream processing
- Objective marker: improve input quality before inference by selecting frames with stronger geometric usefulness
- Objective marker: normalize reconstructed geometry to a consistent real-world scale
- Objective marker: avoid duplicating preprocessing and instead branch from one scaled reconstruction into two outputs
- Objective marker: support both visualization-oriented and geometry-oriented downstream uses
- Objective marker: expose parameters through a desktop interface rather than requiring ad hoc code edits
- Objective marker: preserve experiment-to-production traceability by auditing experiment code before moving it into `src/`
- Objective marker: keep metadata about settings, warnings, counts, and reports for reproducibility
- Evidence:
  - [src/EXPERIMENTS_AUDIT.md](/D:/GitRepos/Thesis/src/EXPERIMENTS_AUDIT.md)
  - [src/pipeline/fast3r_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/fast3r_pipeline.py)
  - [src/app/main_window.py](/D:/GitRepos/Thesis/src/app/main_window.py)
- Missing:
  - explicit prioritized requirements list in thesis wording
  - a short "non-goals" subsection could help later, for example no full metric SLAM, no real-time reconstruction during capture
  - controlled cross-tool benchmark against dense COLMAP MVS and fully optimized Gaussian-splat applications

## Application context

- Current application form: local desktop app / wrapper around Fast3R
- Main workflow:
  - choose video
  - run selected pipeline in background
  - write scene outputs under `src/outputs/scene_<timestamp>/`
  - register result in [scenes_index.json](/D:/GitRepos/Thesis/src/data/scenes_index.json)
  - visualize result in local viewer
- Intended use contexts:
  - indoor scene visualization
  - game-engine-ready intermediate assets
  - simulation preparation
  - later semantic or physics extension
- Input context:
  - handheld indoor video, typically panning / moving through a room
  - strong redundancy and varying quality across frames
- Output context:
  - Gaussian splat scene for view synthesis / high-quality rendering
  - mesh output for engines and geometry-oriented workflows
- Evidence:
  - [src/README.md](/D:/GitRepos/Thesis/src/README.md)
  - [src/pipeline/fast3r_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/fast3r_pipeline.py)
  - [src/viewer/scene_viewer.py](/D:/GitRepos/Thesis/src/viewer/scene_viewer.py)
- Missing:
  - explicit user persona or application scenario
  - thesis-ready screenshot set of the app UI and the viewer

## Structure of the document

- Chapter mapping marker:
  - Chapter 2 should explain Fast3R, multiview reconstruction, Gaussian splats, meshing, scaling, and tool choices
  - Chapter 3 should explain video/frame data, input diagnostics, selection, filtering, and scale normalization inputs
  - Chapter 4 should separate actual inference use from optional fine-tuning preparation
  - Chapter 5 should explain software architecture, module boundaries, UI, and scene output flow
  - Chapter 6 should present run summaries, retention statistics, timings, failures, and output quality tradeoffs
  - Chapter 7 should stay honest about what was achieved versus what remained experimental
- Missing:
  - final sentence-level chapter summaries in author voice

# 2. Theoretical Background

## Brief intro to technologies and ML techniques used

- Core model marker: Fast3R
  - official repo included locally under `fast3r/`
  - positioned as large-scale multiview reconstruction in one forward pass
  - based on DUSt3R / Spann3R lineage
  - local source anchor: [fast3r/README.md](/D:/GitRepos/Thesis/fast3r/README.md)
- Reconstruction theory marker:
  - multiview geometry and cross-view consistency are central, unlike per-frame monocular depth
  - camera pose estimation is derived after model prediction using PnP-like estimation in the local pipeline
  - local anchor: [src/pipeline/fast3r/reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/reconstruction.py)
- Frame selection theory marker:
  - quality heuristics: sharpness via Laplacian variance, entropy, exposure/clipping, brightness balance, ORB texture
  - redundancy control: thumbnail cosine similarity and temporal coverage
  - geometry-aware extension: confidence coverage, point spread, pose validity, camera separation, view-angle change, overlap proxy
  - local anchor: [src/pipeline/fast3r/frame_selection.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/frame_selection.py)
- Scale theory marker:
  - learned reconstruction may remain in arbitrary scene units
  - practical pipeline needs conversion to meters
  - preferred method: metadata-derived scale if available
  - fallback method: geometric reference using real-world dimension / measured dimension
  - local anchor: [src/pipeline/fast3r/scale_normalization.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/scale_normalization.py)
- Point cloud cleanup marker:
  - finite-value filtering
  - radius-percentile outlier filtering
  - optional statistical outlier removal
  - voxel downsampling
  - local anchor: [src/pipeline/fast3r/postprocess.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/postprocess.py)
- Gaussian splatting marker:
  - point-based radiance representation
  - initialized from scaled point cloud
  - positions, colors, normals, local scale estimates, SH coefficients, opacity
  - current implementation initializes DC-only SH and identity rotations unless surface alignment is enabled
  - local anchor: [src/pipeline/fast3r/gaussian_splats.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/gaussian_splats.py)
- Mesh theory marker:
  - Poisson surface reconstruction on oriented points
  - density trimming, connected component cleanup, Laplacian smoothing, decimation
  - optional future comparison space includes Screened Poisson / BPA / advancing front
  - local anchor: [src/pipeline/fast3r/mesh_reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/mesh_reconstruction.py)
- UI/tooling marker:
  - PyQt-based desktop wrapper
  - Open3D for geometry processing
  - PyTorch for Fast3R inference
  - YAML configuration and JSON metadata
- Good theory sources already collected locally:
  - [mesh_reconstruction_from_ply_sources_2026-04-06.md](/D:/GitRepos/Thesis/thesis_docs/research_notes/mesh_reconstruction_from_ply/mesh_reconstruction_from_ply_sources_2026-04-06.md)
- Missing:
  - short formal explanation of DUSt3R/Spann3R relationship from primary papers
  - exact citation export for Fast3R, DUSt3R, Spann3R, Open3D

## Similar tools or existing systems

- Comparison marker: COLMAP SfM
  - strengths: geometric consistency, camera recovery, clean sparse geometry
  - weakness in this project context: sparse output not enough for dense scene reuse
  - evidence: [RESULTS_SUMMARY.md](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md)
- Comparison marker: MiDaS monocular depth pipeline
  - strengths: fast dense output
  - weaknesses: per-frame inconsistency, scale ambiguity, poor geometry
  - evidence: same file as above
- Comparison marker: raw visual-only frame selection
  - cheaper and simpler
  - weaker adaptation to actual multiview geometry
  - evidence: [experiments/input_quality_lab/README.md](/D:/GitRepos/Thesis/experiments/input_quality_lab/README.md), [prefilter_summary.json](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/prefilter_summary.json)
- Comparison marker: geometry-aware frame selection
  - uses Fast3R itself as a cheap probe before the final reconstruction
  - creates clear ablation story for thesis
  - evidence: [IMPLEMENTATION_NOTES.txt](/D:/GitRepos/Thesis/experiments/geometry_test/IMPLEMENTATION_NOTES.txt)
- Comparison marker: Gaussian versus mesh outputs
  - Gaussian output favors rendering and visual fidelity
  - mesh output favors topology, geometry exchange, engine compatibility
- Application-positioning marker:
  - the desktop application is not merely a Fast3R wrapper; it is designed as a fast local feedback workflow from video to inspectable scene
  - compared with multi-stage reconstruction and separately optimized 3DGS workflows, the application integrates frame curation, Fast3R inference, shared geometry processing, Gaussian initialization, mesh export, scene history, and local viewing
  - the user-facing differentiator is reduced time-to-inspectable-scene
- External workflow comparison marker: COLMAP SfM / MVS
  - official dense workflow includes sparse reconstruction, undistortion, stereo depth and normal estimation, fusion, and optional meshing
  - archived internal sparse-only baseline already reported approximately 8 minutes for 100 sampled frames
  - official source: [COLMAP tutorial](https://colmap.github.io/tutorial.html)
- External workflow comparison marker: Nerfstudio Splatfacto
  - custom video processing uses COLMAP and FFmpeg before `ns-train splatfacto`
  - trained splat export occurs after the optimization workflow
  - official sources: [Nerfstudio custom-data guide](https://docs.nerf.studio/quickstart/custom_dataset.html), [Splatfacto guide](https://docs.nerf.studio/nerfology/methods/splat.html)
- External workflow comparison marker: original INRIA 3DGS implementation
  - prepares COLMAP-ready SfM inputs and runs a PyTorch Gaussian optimizer
  - reference configuration defaults to 30,000 training iterations
  - official source: [INRIA Gaussian Splatting reference implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- External workflow comparison marker: Jawset Postshot
  - supports image or video input, computes camera poses and sparse points when they are absent, and then starts radiance-field training
  - official documentation explicitly describes camera tracking as a multi-step process that takes time before training begins
  - official sources: [Postshot importing guide](https://www.jawset.com/docs/d/Postshot%2BUser%2BGuide/Importing%2BImages), [Postshot training configuration](https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Interface/Training%2BConfiguration)
- Comparison honesty marker:
  - this is primarily a workflow-shape comparison
  - do not claim identical outputs: the application currently exports Gaussian initialization data, while full 3DGS applications spend additional time optimizing a trained radiance-field representation
  - do not assign unmeasured runtime numbers to Nerfstudio, INRIA 3DGS, or Postshot
- Missing:
  - thesis should include a compact table: COLMAP vs Nerfstudio Splatfacto vs INRIA 3DGS vs Postshot vs this application
  - controlled local cross-tool runtime benchmark remains future work

## Justification for chosen tools (e.g., FastAPI, Streamlit, TensorFlow)

- Chosen model: Fast3R
  - reason marker: practical multiview learned reconstruction with dense point output and confidence signals
  - reason marker: better suited than per-frame depth for geometry consistency
  - reason marker: easier to integrate into a desktop reconstruction workflow than a full traditional SfM+MVS stack for this project
- Chosen geometry library: Open3D
  - reason marker: provides point cloud IO, normals, Poisson reconstruction, component cleanup, decimation, and metric checks in one local toolkit
- Chosen app framework: PyQt
  - reason marker: desktop-native control panel for running local inference and exposing parameters without a web deployment step
- Chosen configuration style: YAML + JSON reports
  - reason marker: readable settings, traceable run metadata, thesis-friendly audit trail
- Chosen output design: branch after shared scaling and cleanup
  - reason marker: avoid duplicated preprocessing and keep both output types geometrically consistent
- Chosen postprocess strategy:
  - radius-percentile filtering preferred from experiments
  - Poisson preferred as primary mesh method because it is robust to noise and tends to fill small holes
  - Gaussian initialization based on scaled point cloud because it preserves dense learned geometry for rendering
- Evidence:
  - [src/EXPERIMENTS_AUDIT.md](/D:/GitRepos/Thesis/src/EXPERIMENTS_AUDIT.md)
  - [mesh_reconstruction_from_ply_sources_2026-04-06.md](/D:/GitRepos/Thesis/thesis_docs/research_notes/mesh_reconstruction_from_ply/mesh_reconstruction_from_ply_sources_2026-04-06.md)
- Missing:
  - explicit rationale paragraph for why not keep COLMAP-only or MiDaS-only approach
  - citation-backed wording for Gaussian splatting if the thesis includes theory depth there

# 3. Dataset and Preprocessing

## Data origin, format, and structure

- Main project data marker:
  - input is video rather than a supervised training dataset
  - representative source video referenced throughout experiments: `data/raw/irl_room_video_2.mp4`
- Example scan metadata from geometry-aware experiments:
  - total frames: 10,428
  - fps: about 60.1
  - original resolution: 848 x 478
  - evidence: [geometry_selection_summary.json](/D:/GitRepos/Thesis/experiments/geometry_test/results/irl_room_video_2_run_01/geometry_selection_summary.json)
- App-side scene output structure marker:
  - each run stored in `src/outputs/scene_<timestamp>/`
  - includes metadata, scale report, point retention report, Gaussian report, optional mesh report
  - evidence: [scene_metadata.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/scene_metadata.json)
- Fine-tuning dataset marker:
  - training support is being prepared separately in `experiments/vast_training`
  - ARKitScenes is the current preferred public dataset in that bundle
  - some files still mention ScanNet as older primary path, so thesis must present this as evolving prep work, not a finalized training dataset section
- Missing:
  - explicit dataset inventory table for all captured videos used in experiments
  - if thesis wants supervised data subsection, list of exact fine-tuning scenes still needed

## Cleaning, normalization, labeling

- Pre-inference cleaning marker:
  - scan video with stride
  - compute candidate frame metrics
  - reject low-quality and near-duplicate frames
  - optionally run geometry-aware probe pass before final selection
- Candidate frame metrics available in code:
  - Laplacian sharpness
  - grayscale entropy
  - brightness mean
  - clipped ratio
  - ORB keypoints
  - thumbnail similarity
- Input-quality lab marker:
  - used to diagnose blur, clipping, temporal overlap, optional coverage
  - artifacts: [input_quality_lab/README.md](/D:/GitRepos/Thesis/experiments/input_quality_lab/README.md)
- Example prefilter results:
  - scan stride 15
  - candidate frames 696
  - target frames 80
  - mean selected score about 0.956
  - evidence: [prefilter_summary.json](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/prefilter_summary.json)
- Geometry-aware two-stage selection marker:
  - stage 1 visual shortlist
  - stage 2 low-resolution Fast3R probe
  - greedy selection using geometry quality, diversity, overlap, temporal coverage
  - evidence: [IMPLEMENTATION_NOTES.txt](/D:/GitRepos/Thesis/experiments/geometry_test/IMPLEMENTATION_NOTES.txt)
- Post-inference normalization marker:
  - confidence thresholding
  - quantile-based confidence retention
  - weak-texture retention option
  - per-view rejection based on p50 / p90 confidence statistics
  - numeric sanity filter and outlier removal
- Scale normalization marker:
  - mandatory before output branching
  - isotropic global scaling
  - metadata-first, geometry-reference fallback
  - evidence: [scale_normalization.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/scale_normalization.py)
- Labeling marker:
  - no manual semantic labeling in main pipeline
  - if chapter wording says "labeling," note that this thesis used unsupervised / self-consistent reconstruction outputs rather than labeled semantic classes
- Missing:
  - exact before/after image examples of rejected and retained frames
  - explicit quality flag thresholds table gathered into one thesis table

## Train-test split, augmentation (if any)

- Main reconstruction pipeline marker:
  - not a supervised training pipeline
  - no train-test split for the deployed application inference path
- Fine-tuning prep marker:
  - training configs exist under [experiments/vast_training/configs](/D:/GitRepos/Thesis/experiments/vast_training/configs)
  - current `vast_config.json` defaults:
    - dataset `arkitscenes`
    - 6 views
    - image size 512
    - batch size 1
    - bf16 mixed precision
    - max steps 12,000
  - evidence: [vast_config.json](/D:/GitRepos/Thesis/experiments/vast_training/configs/vast_config.json)
- Fine-tuning augmentation marker:
  - current config exposes `aug_crop`
  - this belongs to training-prep chapter content, not main implemented application results
- Fine-tuning data-preparation marker:
  - raw public dataset path currently prepared around ARKitScenes
  - raw download must be transformed into `arkitscenes_processed`
  - expected processed structure:
    - `Training/all_metadata.npz`
    - `Test/all_metadata.npz`
    - per-scene folders with `vga_wide`, `lowres_depth`, `lowres_wide.traj`, `vga_wide_intrinsics`
  - preparation script:
    - [prepare_arkitscenes_processed.py](/D:/GitRepos/Thesis/experiments/vast_training/prepare_arkitscenes_processed.py)
  - implementation notes:
    - converts trajectory lines to pose matrices
    - resolves per-frame intrinsics from `.pincam`
    - generates `all_metadata.npz` with scenes, scene ids, image names, intrinsics, trajectories, and frame pairs
    - maps Apple `Validation/` split into local `Test/` split expected by loader
- Fine-tuning multiview sampling marker:
  - launcher targets `ARKitScenes_Multiview`
  - training set uses configurable:
    - `num_views`
    - `window_size`
    - `num_samples_per_window`
    - `ordered`
    - `aug_crop`
  - validation uses smaller deterministic sampling with `ordered=true` and fixed seed
- Fine-tuning split/evidence marker:
  - training and validation logic is encoded in launchers even if full runs are not yet completed
  - code anchors:
    - [launch_training.py](/D:/GitRepos/Thesis/experiments/vast_training/launch_training.py)
    - [launch_evaluation.py](/D:/GitRepos/Thesis/experiments/vast_training/launch_evaluation.py)
- Missing:
  - confirmed final training/validation split actually executed
  - completed fine-tuning logs and evaluation outputs
  - if no fine-tuning is finished, chapter should clearly say "training support prepared but not fully executed within project window"

# 4. ML Model Design and Training (If exists)

## Architecture used (e.g., Random Forest, CNN, LSTM, Transformers)

- Core inference architecture marker:
  - Fast3R multiview reconstruction model
  - transformer-based lineage through DUSt3R / Spann3R ecosystem
  - local implementation is consumed through [reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/reconstruction.py)
- Pipeline-level architecture marker:
  - frame selection
  - Fast3R inference
  - confidence retention
  - scale normalization
  - shared point cloud processing
  - Gaussian branch
  - mesh branch
- Experimental training architecture marker:
  - only GA/global head intended for fine-tuning
  - backbone intended frozen
  - local head intended frozen
  - evidence: [experiments/vast_training/README.md](/D:/GitRepos/Thesis/experiments/vast_training/README.md), [head_finetune_callback.py](/D:/GitRepos/Thesis/experiments/vast_training/head_finetune_callback.py)
- GA head technical marker:
  - in the local training bundle, "GA head" is mapped to the Fast3R global prediction head
  - trainable token for this stage is `downstream_head`
  - local head token is `downstream_head_local`
  - stage variants supported by launcher:
    - `ga_head`
    - `ga_head_plus_local`
    - `decoder_plus_ga_head`
    - `local_head`
  - current thesis-preferred stage:
    - `ga_head`
  - rationale marker:
    - lowest-risk adaptation path for indoor global consistency
    - avoids full backbone retraining
- Freeze strategy marker:
  - implemented by callback, not by manual one-off script edits
  - callback iterates `net.named_parameters()`
  - enables gradient updates only for names matching requested substrings
  - raises explicit error if zero trainable parameters remain
  - evidence: [head_finetune_callback.py](/D:/GitRepos/Thesis/experiments/vast_training/head_finetune_callback.py)
- Fine-tuning system architecture marker:
  - dataset preparation layer:
    - raw ARKitScenes download
    - processed metadata generation
  - launch/control layer:
    - JSON profile + environment variable overrides
  - training layer:
    - Fast3R Lightning training entrypoint
    - parameter-freezing callback
    - VRAM monitor callback
  - evaluation layer:
    - validation launcher against selected dataset split
  - export layer:
    - checkpoint conversion and final export back to HF-style format
- Thesis caution marker:
  - architecture explanation can be strong for the inference pipeline
  - fine-tuning architecture should be framed as prepared infrastructure unless trained checkpoints and evaluation exist
- Missing:
  - a clean architecture figure for the fine-tuning setup if this stays in the thesis

## Hyperparameter tuning

- Frame selection defaults selected from experiment audit:
  - `target_frames = 80`
  - `selection_mode = geometry_aware`
  - `visual_shortlist_target = 120`
  - `probe_max_frames = 96`
  - `scan_stride = 15`
  - `shortlist_multiplier = 3`
  - `dedupe_similarity = 0.985`
  - `min_frame_gap = 120`
  - `probe_image_size = 256`
  - `probe_dtype = float32`
  - `probe_confidence_threshold = 1.0`
  - `probe_pnp_iters = 50`
  - evidence: [src/EXPERIMENTS_AUDIT.md](/D:/GitRepos/Thesis/src/EXPERIMENTS_AUDIT.md)
- App-side active default profile later evolved toward higher detail:
  - `target_frames = 180`
  - `scan_stride = 10`
  - `dtype = bfloat16`
  - `min_confidence_threshold = 1.05`
  - `confidence_keep_ratio = 0.85`
  - `view_conf_p50_min = 1.2`
  - `view_conf_p90_min = 1.45`
  - `voxel_size = 0.001`
  - evidence: [settings.yaml](/D:/GitRepos/Thesis/src/config/settings.yaml)
- Mesh parameter markers:
  - Poisson depth range seen in configs and outputs: 9 to 10
  - trim quantile around `0.01` or lower
  - component filtering and decimation strongly affect final real-time usability
- Gaussian parameter markers:
  - scale clamping
  - density-driven opacity
  - optional surface alignment
- Fine-tuning config marker:
  - ARKitScenes default `num_views = 6` to fit time/VRAM constraints
  - OOM fallback prepared: drop to 4 views, image size 384
- Fine-tuning profile marker:
  - available config profiles:
    - `quick_debug`
    - `arkitscenes_quick_debug`
    - `scannet_3day_finetune`
    - `final_run`
    - `vast_config`
  - current preferred final rental-safe profile:
    - [vast_config.json](/D:/GitRepos/Thesis/experiments/vast_training/configs/vast_config.json)
- Current `vast_config` parameter markers:
  - task name: `vast_finetune_arkitscenes`
  - run name: `vast_ga_head_arkitscenes`
  - stage: `ga_head`
  - devices: 1
  - training:
    - `num_views = 6`
    - `image_size = 512`
    - `batch_size = 1`
    - `batch_size_val = 1`
    - `num_workers = 8`
    - `num_workers_val = 2`
    - `precision = bf16-mixed`
    - `lr = 1.5e-5`
    - `max_steps = 12000`
    - `accumulate_grad_batches = 2`
    - `checkpoint_every_n_epochs = 1`
    - `early_stopping_patience = 8`
    - `log_every_n_steps = 10`
    - `val_check_interval_steps = 500`
  - dataset:
    - `name = arkitscenes`
    - `root = /workspace/datasets/arkitscenes_processed`
    - train sample count `16000`
    - val sample count `200`
    - train `num_samples_per_window = 5`
    - val `num_samples_per_window = 1`
  - OOM fallback:
    - enabled
    - batch size 1
    - views 4
    - image size 384
- Parameter-override marker:
  - launchers allow env or CLI override for:
    - `DATASET_NAME`
    - `SCANNET_ROOT`
    - `ARKITSCENES_ROOT`
    - `OUTPUT_DIR`
    - `CHECKPOINT_DIR`
    - `PRETRAINED_FAST3R_CKPT`
    - `NUM_VIEWS`
    - `IMAGE_SIZE`
    - `BATCH_SIZE`
    - `LR`
    - `MAX_STEPS`
    - `RESUME_CKPT`
- Resolution strategy marker:
  - launcher converts a single target image size into a list of aspect-ratio-compatible resolutions
  - this is a useful implementation detail for thesis code explanation because it shows practical handling rather than hard-coded one-size inputs
- Missing:
  - one thesis table collecting all final chosen parameters in one place
  - explicit ablation writeup on why final 180-frame app preset differs from earlier 80-frame experiment defaults
  - completed tuned training run from which to justify final hyperparameters empirically

## Training setup and environment (e.g., Google Colab, GPU used)

- Main reconstruction environment marker:
  - local desktop execution
  - PyTorch + local Fast3R repo
  - evidence of ROCm/HIP memory messages in archived runs suggests AMD GPU testing occurred for some experiments
- Concrete runtime evidence from dual-output experiments:
  - successful runs recorded GPU peak memory roughly 11.2 to 17.4 GB during inference
  - OOM examples on 15.92 GiB GPU capacity are logged in [run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv)
  - example failure: trying to allocate 20.25 GiB
- Fine-tuning environment marker:
  - prepared for Vast.ai rental environment
  - shell scripts handle upload, setup, dataset prep, training, evaluation, export, and download
  - minimal remote layout documented in [experiments/vast_training/README.md](/D:/GitRepos/Thesis/experiments/vast_training/README.md)
- Fine-tuning remote layout marker:
  - expected remote tree:
    - `/workspace/fast3r`
    - `/workspace/vast_training`
    - `/workspace/datasets/arkitscenes_raw`
    - `/workspace/datasets/arkitscenes_processed`
    - `/workspace/checkpoints`
    - `/workspace/logs`
- Fine-tuning execution flow marker:
  - upload only `fast3r/` and `vast_training/`
  - install environment and convert base checkpoint
  - download raw ARKitScenes
  - prepare `arkitscenes_processed`
  - run quick debug
  - run final training
  - evaluate checkpoint
  - export final checkpoint
  - download run artifacts back locally
- Fine-tuning script anchors:
  - [vast_setup_env.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_setup_env.ksh)
  - [vast_download_arkitscenes.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_download_arkitscenes.ksh)
  - [vast_prepare_arkitscenes.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_prepare_arkitscenes.ksh)
  - [vast_quick_debug.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_quick_debug.ksh)
  - [vast_train_final_run.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_train_final_run.ksh)
  - [vast_resume_training.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_resume_training.ksh)
  - [vast_evaluate_run.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_evaluate_run.ksh)
  - [vast_export_final_checkpoint.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_export_final_checkpoint.ksh)
- Training-command generation marker:
  - the thesis can show that launcher scripts do not hard-code one training line
  - instead they generate the Fast3R Lightning/Hydra command from JSON profiles plus environment overrides
  - this is good engineering evidence for reproducibility and rerun control
- VRAM/risk-control marker:
  - reduced to 6 views specifically to stay near a 12-hour rental budget on a 48 GB class GPU
  - OOM fallback path exists in config
  - separate quick-debug profile exists to validate dataset/path correctness before spending on a full run
- Thesis caution:
  - unless logs/checkpoints show completed training, this section should separate "implemented and executed inference environment" from "prepared training environment"
- Missing:
  - exact local GPU model and CPU/RAM for the final application experiments
  - completed fine-tuning runtime logs
  - actual rented Vast host specification used for the final thesis run

## Fine-tuning completion markers for later thesis writing

- This subsection is intentionally procedural and should later be converted into thesis evidence only after execution
- ############### FINE-TUNE PART 1
- Date marker:
  - active Vast.ai fine-tuning/debugging session on 2026-05-26
- Scope marker:
  - goal was to fine-tune only the Fast3R GA/global head on ARKitScenes
  - target stage token remained `downstream_head`
- Executed-debug marker:
  - `arkitscenes_quick_debug` run executed successfully through actual train/validation loops
  - this proved dataset preparation, callback wiring, VRAM logging, and Lightning launch flow
  - it was only a smoke test, not the final run, because it used tiny batch limits and only a handful of effective optimization steps
- Important fixes made during this session:
  - disabled built-in Lightning `Trainer.test()` after training by forcing `test=False` in the generated Hydra command
  - fixed launcher bug where `limit_train_batches=1.0` was incorrectly treated like `1` and collapsed `val_check_interval` to `1`; corrected behavior now preserves `val_check_interval=500` or other configured values for full-epoch runs
  - confirmed remote execution must explicitly use:
    - `PYTHON_BIN=/workspace/venvs/fast3r-py312/bin/python`
    - `PYTHONPATH=/workspace`
  - otherwise custom callbacks under `vast_training.*` fail to import on the remote machine
- Runtime-control marker:
  - a detached `nohup` launch flow was adopted so training can continue outside Jupyter
  - log-following with `tail -f` is normal and does not mean the job is frozen
- Time-budget marker:
  - the original `vast_config` profile was judged too long for the remaining rental budget
  - a shorter profile was introduced:
    - [arkitscenes_10hour.json](/D:/GitRepos/Thesis/experiments/vast_training/configs/arkitscenes_10hour.json)
  - key changes:
    - `max_steps = 4000`
    - `val_check_interval_steps = 2000`
    - validation sample count reduced to `50`
- Artifact-location marker:
  - training metrics are written under `/workspace/logs/.../csv/version_0/metrics.csv`
  - checkpoints are written under `/workspace/logs/.../checkpoints/`
  - download-friendly copies should also be placed under `/workspace/TO_BE_DOWNLOADED_TO_LOCAL_MACHINE/...` for later thesis figures and discussion
- Honesty marker:
  - this part of the thesis can now be written as "fine-tuning infrastructure implemented, debugged, launched, and completed for one shortened ARKitScenes GA-head run"
  - do not write that a controlled pretrained-versus-fine-tuned comparison was completed unless `comparison_summary.json`, `comparison_metrics.csv`, `pretrained_metrics.csv`, and `finetuned_metrics.csv` are later found or regenerated
- Completion marker:
  - a real detached Vast.ai run completed on 2026-05-27 for the shortened ARKitScenes profile
  - profile:
    - [arkitscenes_10hour.json](/D:/GitRepos/Thesis/experiments/vast_training/configs/arkitscenes_10hour.json)
  - run name:
    - `arkitscenes_10hour_ga_head`
  - observed completion evidence from remote log:
    - epoch 0 finished at `16000/16000`
    - wall-clock training time about `6:32:49`
    - final logged `val/loss` about `0.0755`
    - final logged `trainer/lr` about `4.534e-06`
  - artifact markers from remote log:
    - selected checkpoint path:
      - `/workspace/logs/vast_finetune_arkitscenes/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt`
    - metrics CSV path:
      - `/workspace/logs/vast_finetune_arkitscenes/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv`
  - post-run issue marker:
    - bundling failed after successful training because `bundle_results.py` attempted to copy `train_log.txt` onto itself and raised `SameFileError`
    - this did not invalidate the completed training run or the written checkpoint
- Collected local artifacts:
  - final run metrics CSV:
    - [metrics.csv](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv)
  - final run hyperparameter snapshot:
    - [hparams.yaml](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/hparams.yaml)
  - final run config tree:
    - [config_tree.log](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/config_tree.log)
  - final run training log:
    - [vast_finetune_arkitscenes.log](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/vast_finetune_arkitscenes.log)
  - checkpoint directories:
    - [last.ckpt](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt)
    - [epoch_000.ckpt](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints/epoch_000.ckpt)
- Required artifacts still to collect or regenerate:
  - downloaded dataset subset description
  - processed dataset manifest
  - exported final checkpoint path, if application-side fine-tuned inference is claimed
  - paired pretrained-versus-fine-tuned evaluation files:
    - `comparison_summary.json`
    - `comparison_metrics.csv`
    - `pretrained_metrics.csv`
    - `finetuned_metrics.csv`
  - GPU/VRAM/runtime note in a clean table
- Required analysis questions once run is complete:
  - did GA-head-only tuning improve indoor global alignment qualitatively
  - did wall/plane stability improve
  - did pose consistency across views improve
  - did low-texture failure cases reduce
  - did point retention confidence distributions shift
  - did final reconstruction quality improve on the same personal videos as baseline
- Required comparison protocol:
  - first controlled quantitative checkpoint comparison on the same public split:
    - pretrained `/workspace/checkpoints/fast3r_vit_large_hf_as_lightning.ckpt`
    - fine-tuned `/workspace/logs/vast_finetune_arkitscenes/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt`
    - same ARKitScenes validation split as the shortened training profile
    - same remote machine family and same evaluation launcher settings
  - implementation marker:
    - paired evaluation helper:
      - [launch_evaluation_compare.py](/D:/GitRepos/Thesis/experiments/vast_training/launch_evaluation_compare.py)
    - remote shell wrapper:
      - [vast_run_evaluation_compare.ksh](/D:/GitRepos/Thesis/experiments/vast_training/vast_run_evaluation_compare.ksh)
    - comparison artifacts should include:
      - `comparison_summary.json`
      - `comparison_metrics.csv`
      - `pretrained_metrics.csv`
      - `finetuned_metrics.csv`
  - baseline pretrained Fast3R on selected indoor videos
  - fine-tuned Fast3R on same videos
  - compare:
    - frame selection behavior if unchanged
    - point retention
    - scale-normalized point cloud quality
    - Gaussian visual quality
    - mesh artifact profile
    - runtime / memory cost
- If time is too short for a full metric study:
  - at minimum capture a controlled before/after comparison on 2 to 3 room videos with the same app settings
  - save identical report bundles for baseline and fine-tuned runs

# 5. App Integration & Design (If exists)

## Application as a software contribution

- Thesis positioning marker:
  - the application is a first-class contribution of the thesis, not only a thin launcher for Fast3R
  - it transforms a research model into a desktop workflow that starts from a monocular indoor video and ends with an inspectable scene, stored metadata, and reusable output artifacts
  - Chapter 5 should therefore describe both the reconstruction pipeline and the user-facing software system
- Current application name / concept:
  - Fast3R Indoor Reconstruction Desktop Application
  - user-facing concept: 3D Room Reconstruction Tool
- Current maturity:
  - alpha / research prototype with a productionized desktop wrapper
  - functional enough to demonstrate scene creation, reconstruction execution, output storage, and embedded inspection
  - not yet a full commercial project-management tool
- Stack:
  - Python
  - PyQt6
  - PyQt6-WebEngine / QWebEngineView
  - Fast3R
  - PyTorch
  - OpenCV
  - Open3D
  - Viser
  - NumPy
  - YAML and JSON for configuration and reports
- Application-level contribution:
  - scene-first workflow instead of treating videos as the only primary object
  - local scene library backed by JSON records
  - compact reconstruction properties panel
  - selectable Fast3R weights profile: Default Fast3R or Fine-Tuned Fast3R
  - automatic preset mode with manual advanced controls only when needed
  - background reconstruction worker to keep the UI responsive
  - workflow menu and view menu, including restore actions for movable/closable docks
  - embedded Viser viewer inside the PyQt6 workspace
  - persistent output metadata and reports for reproducibility
- Performance contribution inside the software design:
  - the UI and pipeline are built around reducing time-to-inspectable-scene
  - Open Output remains accessible and guides the user if outputs have not yet been constructed
  - mesh is optional and disabled in the default high-quality preset to avoid unnecessary runtime in the normal feedback loop
  - Gaussian and mesh outputs reuse the same processed reconstruction rather than running duplicate reconstruction stages

## Current user workflow

- Workflow as implemented:
  - launch desktop application
  - use Workflow -> New Scene
  - configure:
    - scene name
    - source video
    - optional description
    - pipeline
    - weights profile: Default Fast3R or Fine-Tuned Fast3R
    - reconstruction preset
    - optional tags
  - select a scene from Scene Library
  - optionally run Analyze Video to produce a temporary calculated preset
  - run reconstruction
  - monitor progress in the bottom pipeline tracker
  - open output in the embedded Viser viewer
- Scene Library:
  - visible columns: Scene, Status, Created, Pipeline, Reconstruction
  - scene name column stretches and elides when the dock is narrow
  - user-defined scene name is the primary label, not the generated internal identifier
  - metadata remains available through row data and tooltips
- Reconstruction Properties panel:
  - Input: video filename with full path as tooltip and Change action
  - Pipeline: pipeline selector and weights selector
  - Auto Configuration: Auto preset checkbox and preset dropdown
  - Status: compact next-step message
  - Actions: Analyze, Run, Open Output
  - manual parameter tabs appear only when Auto preset is disabled
- Open Output behavior:
  - always available when no pipeline is actively running
  - if no scene/output exists, it shows a message instructing the user to construct outputs by pressing Run Reconstruction
  - if output exists, it starts/reuses the local Viser server and loads the URL into QWebEngineView
- Figures to insert:
  - `[FIGURA 5.2 - Fluxul UI scene-first al aplicatiei]`
  - `[FIGURA 5.4 - Interfata principala a aplicatiei desktop]`
  - `[FIGURA 5.5 - Dialogul New Scene si selectia profilului de greutati]`
  - `[FIGURA 5.6 - Scene Library si panoul Reconstruction Properties]`
  - `[FIGURA 5.7 - Viewerul Viser incorporat pentru scena reconstruita]`

## Current module responsibilities

| Module | Responsibility in thesis wording | Important implementation details |
| --- | --- | --- |
| `src/app/main_window.py` | Main desktop window and workflow orchestration | builds menus, docks, reconstruction panel, embedded viewer, action state, run config |
| `src/app/new_scene_dialog.py` | Scene creation workflow | captures name, video, description, pipeline, weights profile, preset, tags |
| `src/app/scene_list_widget.py` | Scene Library | shows local scenes, preserves selected scene data, adaptive table columns |
| `src/app/embedded_viewer.py` | Embedded output viewer widget | wraps QWebEngineView and exposes load/clear/reload state |
| `src/app/stage_progress_widget.py` | Pipeline progress tracker | displays Frames, Fast3R, Geometry, Output stages and progress text |
| `src/app/worker.py` | Background execution | runs pipeline in QThread and emits progress/success/failure |
| `src/app/theme.py` | Central desktop dark theme | semantic color tokens and one generated stylesheet |
| `src/application/scene_repository.py` | Scene persistence | reads/writes `src/data/scenes_index.json` |
| `src/application/viewer_service.py` | Viewer launch policy | starts/reuses local viewer route without forcing external browser |
| `src/models/scene.py` | User-facing scene entity | stores scene name, video, pipeline, model profile, preset, status and metadata |
| `src/models/scene_result.py` | Pipeline output DTO | stores output dir, pointcloud, mesh, camera poses and metadata |
| `src/pipeline/pipeline_factory.py` | Pipeline registry | maps `default` to `Fast3rPipeline` |
| `src/pipeline/fast3r_pipeline.py` | Pipeline orchestrator | frame selection, Fast3R run, filtering, scale, Gaussian, mesh, metadata |
| `src/pipeline/fast3r/frame_selection.py` | Frame selection | visual prefiltering and geometry-aware Fast3R probe |
| `src/pipeline/fast3r/reconstruction.py` | Fast3R inference wrapper | loads selected profile with `Fast3R.from_pretrained`, extracts points/confidence/poses |
| `src/pipeline/fast3r/postprocess.py` | Point cloud cleanup | finite filtering, radius-percentile filtering, voxel downsampling, PLY save |
| `src/pipeline/fast3r/scale_normalization.py` | Metric scale conversion | converts arbitrary reconstruction units toward meters |
| `src/pipeline/fast3r/gaussian_splats.py` | Gaussian initialization branch | exports Gaussian PLY/NPZ and report |
| `src/pipeline/fast3r/mesh_reconstruction.py` | Mesh branch | optional Poisson reconstruction, cleanup, smoothing, decimation |
| `src/viewer/scene_viewer.py` | Local viewer process | serves point cloud/Gaussian/mesh via Viser and optional CLI browser behavior |

## Scene and output model

- Important distinction:
  - video is a source input
  - scene is the user-facing domain object
  - output directory is the result of a reconstruction run
- Scene fields:
  - `scene_id`
  - `name`
  - `source_video`
  - `description`
  - `pipeline`
  - `model_profile`
  - `reconstruction_preset`
  - `tags`
  - `status`
  - `created_at`
  - `reconstruction_type`
  - output paths and metadata
- Weight profiles:
  - Default Fast3R: `jedyang97/Fast3R_ViT_Large_512`
  - Fine-Tuned Fast3R: local exported Hugging Face-compatible folder `data/models/fast3r_arkitscenes_ga_head_hf`
  - the selected profile is stored in the scene and passed to both:
    - geometry-aware Fast3R probe
    - main Fast3R reconstruction
  - if the local fine-tuned folder is missing, the UI warns before reconstruction starts
- Output artifacts:
  - `scene_metadata.json`
  - `fast3r_point_retention_report.json`
  - `pointcloud_processing_report.json`
  - `scale_normalization_report.json`
  - `pointcloud.ply`
  - raw arrays such as `raw_points.npy`, `raw_colors.npy`, `poses.npy`
  - `frames/`
  - `gaussian_splat/gaussian_splats_init.ply`
  - `gaussian_splat/gaussian_params.npz`
  - `gaussian_splat/scaled_points_for_gaussian.ply`
  - `gaussian_splat/gaussian_splat_report.json`
  - optional `mesh/mesh_scaled_poisson.ply`
  - optional `mesh/mesh_scaled_poisson.obj`
  - optional `mesh/mesh_branch_report.json`
- Figure to insert:
  - `[FIGURA 5.3 - Structura outputului pentru o scena reconstruita]`

## Application limitations to state honestly

- The app is a local desktop research prototype, not a deployed cloud product.
- There is no full project file format with create/load/save project lifecycle.
- There is no job cancellation yet.
- There is no GLB export in the production app.
- Mesh export exists but is disabled in the active high-quality preset.
- Gaussian export is an initialization/inspectable representation, not a full optimized 3DGS training process.
- MainWindow still owns too much UI and runtime configuration logic.
- Fast3R loading is not yet isolated behind a clean infrastructure adapter.
- Embedded viewer depends on PyQt6-WebEngine and the local Viser server.

## UML diagrams, flowcharts, component diagrams

- Existing diagram candidate:
  - [fast3r_geometry_aware_pipeline.svg](/D:/GitRepos/Thesis/thesis_docs/writing/fast3r_geometry_aware_pipeline.svg)
- New SVG diagrams created for the thesis application chapter:
  - [figura_5_1_arhitectura_aplicatiei_desktop.svg](/D:/GitRepos/Thesis/thesis_docs/document/figures/application/figura_5_1_arhitectura_aplicatiei_desktop.svg)
  - [figura_5_2_fluxul_ui_scene_first.svg](/D:/GitRepos/Thesis/thesis_docs/document/figures/application/figura_5_2_fluxul_ui_scene_first.svg)
  - [figura_5_3_structura_outputului_scene.svg](/D:/GitRepos/Thesis/thesis_docs/document/figures/application/figura_5_3_structura_outputului_scene.svg)
- Recommended diagrams to derive from code:
  - high-level pipeline flow: video -> frame selection -> Fast3R -> scale normalization -> point cloud cleanup -> Gaussian / mesh branches -> metadata / viewer
  - module/component diagram:
    - `main_window.py`
    - `worker.py`
    - `pipeline_factory.py`
    - `fast3r_pipeline.py`
    - `scene_viewer.py`
  - data artifact diagram:
    - input video
    - selected frames
    - retained points
    - scene metadata
    - Gaussian outputs
    - mesh outputs
- Architectural constraint worth showing:
  - "do not import directly from experiments" rule, enforced in [pipeline_factory.py](/D:/GitRepos/Thesis/src/pipeline/pipeline_factory.py) and [base_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/base_pipeline.py)
- Missing:
  - no formal UML diagram file found besides the SVG pipeline figure
  - state/sequence diagram for background execution could help

## Description of key modules or components

- App shell:
  - [src/main.py](/D:/GitRepos/Thesis/src/main.py)
  - launches desktop app
- Main UI:
  - [main_window.py](/D:/GitRepos/Thesis/src/app/main_window.py)
  - parameter tabs for Frames, Model, Scale, Outputs
  - run control and scene list
  - includes parameter auto-estimation from video properties
- Worker/background execution:
  - [worker.py](/D:/GitRepos/Thesis/src/app/worker.py)
- Pipeline registry:
  - [pipeline_factory.py](/D:/GitRepos/Thesis/src/pipeline/pipeline_factory.py)
  - default pipeline is `Fast3rPipeline`
- Main orchestrator:
  - [fast3r_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/fast3r_pipeline.py)
- Fast3R submodules:
  - [video_processor.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/video_processor.py)
  - [frame_selection.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/frame_selection.py)
  - [reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/reconstruction.py)
  - [scale_normalization.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/scale_normalization.py)
  - [postprocess.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/postprocess.py)
  - [gaussian_splats.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/gaussian_splats.py)
  - [mesh_reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/mesh_reconstruction.py)
- Scene model:
  - [scene_result.py](/D:/GitRepos/Thesis/src/models/scene_result.py)
- Viewer:
  - [scene_viewer.py](/D:/GitRepos/Thesis/src/viewer/scene_viewer.py)
- Missing:
  - if thesis needs cleaner component descriptions, generate one table: module / responsibility / main inputs / main outputs

## How the model is used in the app

- Model use marker:
  - selected frames are passed to `Fast3RReconstructor`
  - model outputs dense multiview point predictions and confidence signals
  - local postprocess converts these to one global point set with pose estimates and reports
- Important integration detail:
  - Fast3R is used twice in the overall project history
  - first as final reconstruction model
  - second in experiments as a low-resolution geometry probe for frame selection
- Confidence filtering marker:
  - model output is not used raw
  - min confidence threshold, quantile pruning, and view-level pruning are applied before later stages
- Metadata marker:
  - app persists point retention stats, scale decisions, branch outputs, warnings, and settings used
- Evidence:
  - [reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/reconstruction.py)
  - [fast3r_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/fast3r_pipeline.py)
- Missing:
  - thesis screenshot of UI model settings pane

## Input/output flow (e.g., API, web interface)

- Input flow:
  - local video file
  - optional pre-selected frames directory in some experiment flows
  - app reads config defaults from [settings.yaml](/D:/GitRepos/Thesis/src/config/settings.yaml)
- Internal flow:
  - video metrics -> frame selection -> inference -> confidence retention -> scale normalization -> point cloud cleanup -> output branches -> scene registration
- Output files marker:
  - `scene_metadata.json`
  - `fast3r_point_retention_report.json`
  - `pointcloud_processing_report.json`
  - `scale_normalization_report.json`
  - Gaussian `.ply` and `.npz`
  - optional mesh `.ply` and `.obj`
- Example output location:
  - [scene_20260422_171717](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717)
- Output registration marker:
  - [scenes_index.json](/D:/GitRepos/Thesis/src/data/scenes_index.json)
- Missing:
  - a concise filesystem tree diagram in thesis

## Deployment details (optional: Docker, cloud setup, mobile deployment)

- Current deployment form:
  - local desktop execution
  - no cloud deployment for the app itself
- Experimental remote execution:
  - Vast.ai environment prepared for model fine-tuning only
  - not part of the deployed reconstruction app
- Missing:
  - if deployment section is kept, keep it modest and say local desktop prototype + separate cloud training preparation

# 6. Evaluation and Results

## Metrics and test set performance

- Frame-selection metrics available:
  - candidate frame counts
  - shortlist counts
  - pose-valid probe counts
  - selected frame counts
  - per-frame quality metrics
  - per-pair geometry/overlap metrics
- Reconstruction metrics available:
  - predicted dense points
  - retained points after min confidence
  - retained points after quantile
  - views rejected by confidence
  - confidence percentiles
  - scale factor
  - point counts after cleanup
  - branch-specific artifact counts
- Example app-side quantitative run:
  - [scene_20260422_171717/scene_metadata.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/scene_metadata.json)
  - selected frames: 180
  - candidate frames: 538
  - shortlist size: 522
  - probe pose valid count: 180
  - raw points: 26,264,029
  - final points after cleanup/voxel: 8,146,338
  - scale to meters: 7.365792240649551
- Example point-retention stats from same run:
  - predicted dense points: 35,389,440
  - final extracted points: 26,264,029
  - final retention ratio: 0.7421
  - views rejected by confidence: 23
  - evidence: [fast3r_point_retention_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/fast3r_point_retention_report.json)
- Gaussian branch metrics from same run:
  - output Gaussians: 8,146,338
  - alpha median about 0.655
  - SH initialization `dc_only`
  - evidence: [gaussian_splat_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/gaussian_splat/gaussian_splat_report.json)
- Mesh branch metrics from same run:
  - points after mesh voxel: 2,432,262
  - points used for mesh: 500,000
  - Poisson depth: 9
  - triangles before decimation: 2,872,061
  - target triangles: 180,000
  - final triangles: 179,965
  - evidence: [mesh_branch_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/mesh/mesh_branch_report.json)
- Runtime metrics available in experiment CSV:
  - example 80-frame Gaussian run total time about 148.27 s
  - example 120-frame run total time about 223.68 s
  - example 300-frame run total time about 487.16 s
  - evidence: [run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv)
- Historical performance baseline:
  - archived sparse-COLMAP experiment reported approximately 8 minutes for 100 sampled frames
  - result contained 1,230 sparse points and 24 reconstructed camera poses
  - use this as an engineering motivation marker, not as a direct benchmark against dense COLMAP MVS or fully optimized Gaussian-splat applications
  - evidence: [RESULTS_SUMMARY.md](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md)
- Stage timings available:
  - frame selection
  - inference
  - scale normalization
  - shared pointcloud processing
  - Gaussian branch
  - mesh branch
  - evidence: [stage_timings.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/system/stage_timings.csv)
- Missing:
  - standardized quantitative metric for geometry accuracy against ground truth
  - direct visual quality evaluation rubric for Gaussian outputs

## Error analysis, model limitations

- Scale limitation marker:
  - metadata-based absolute scale often unavailable
  - fallback used geometric room-height assumption
  - example metadata failure reason recorded explicitly in scale report:
    - no explicit meter/unit field or paired baseline distance
  - evidence: [scene_metadata.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/scene_metadata.json)
- Memory limitation marker:
  - some high-detail runs failed from memory pressure
  - example failure: cannot allocate 230 MiB for array of ~30M elements during high-detail processing
  - example GPU OOM messages logged for 15.92 GiB device
  - evidence: [dual_output_summary.json](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/outputs/presentation_180_high_detail/dual_output_summary.json), [run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv)
- Geometry limitation marker:
  - selected views can still be close together; warnings are logged
  - geometry-aware selection initially behaved more like reordering than strong pruning when candidate pool was too small
  - evidence: [IMPLEMENTATION_NOTES.txt](/D:/GitRepos/Thesis/experiments/geometry_test/IMPLEMENTATION_NOTES.txt)
- Mesh limitation marker:
  - final mesh in representative run was not watertight
  - vertex manifold false
  - self intersecting true
  - 32 connected components remained
  - largest component dominated, so most geometry survived but cleanup was still imperfect
- Gaussian limitation marker:
  - current output is initialization/export stage, not a full separate 3DGS training-and-render pipeline
  - thesis must not overstate it as a fully optimized splat training result unless later evidence is added
- Documentation limitation marker:
  - author writing sample document missing from repo
  - fine-tuning has one completed shortened GA-head run with local validation-loss evidence, but paired pretrained-versus-fine-tuned comparison artifacts were not found locally
- Missing:
  - side-by-side failure figure set
  - quantified comparison between coverage-aware and geometry-aware selectors on same scene

## Examples of successful and failed predictions

- Success example marker:
  - `scene_20260422_171717`
  - completed full branch set including Gaussian and mesh reports
  - achieved large retained point cloud and decimated mesh export
- Success example marker:
  - `presentation_180_geometry_retention`
  - useful for showing heavier confidence/view pruning and resulting retained point cloud
  - evidence in [dual_output_summary.json](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/outputs/presentation_180_geometry_retention/dual_output_summary.json)
- Failure example marker:
  - `presentation_180_high_detail`
  - failed because array allocation exceeded available memory after generating ~30M processed points
  - evidence: [dual_output_summary.json](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/outputs/presentation_180_high_detail/dual_output_summary.json)
- Failure example marker:
  - some archived runs hit HIP OOM during inference or later processing
- Useful figure candidates:
  - selected frame distributions from `selected_frame_indices.txt`
  - retained point cloud renders from `scene_*`
  - mesh artifact visual examples from exported OBJ/PLY in mesh folders
- Missing:
  - curated screenshot set for "success / partial success / failure"

# Thesis Generation Evidence Pack (Local Data Audit 2026-06-03)

This section is written for a future LLM that will generate the final thesis text. It records the strongest local evidence, safe claims, metrics, figure ideas, and technology-choice justifications. It should be treated as a thesis evidence packet, not as final prose.

## Core thesis story to preserve

- The project is an applied indoor 3D reconstruction system, not only a model experiment.
- The application starts from ordinary room video and produces inspectable 3D scene artifacts.
- The strongest contribution is the engineering pipeline around Fast3R:
  - input video scanning
  - visual and geometry-aware frame selection
  - Fast3R multiview inference
  - confidence and numeric filtering
  - scale normalization
  - shared point-cloud cleanup
  - Gaussian-splat initialization
  - optional mesh export
  - local scene metadata, history, and viewer support
- The main application-level differentiator is speed: reducing time-to-inspectable-scene compared with workflows that require long iterative reconstruction, mapping, camera-tracking, dense stereo, or radiance-field training stages.
- The thesis should not overclaim:
  - the Gaussian output is an initialization/export from the processed point cloud, not a fully optimized 3DGS training result
  - no complete ground-truth geometry benchmark exists locally
  - no local paired pretrained-versus-fine-tuned comparison bundle was found
  - the GA-head training evidence supports a validation-loss reduction during the completed run, not a fully controlled model-improvement claim against a pretrained baseline

## Data sources parsed for this audit

- App scene outputs:
  - [src/outputs](/D:/GitRepos/Thesis/src/outputs)
  - representative complete run:
    - [scene_20260422_171717](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717)
  - key report files:
    - `scene_metadata.json`
    - `fast3r_point_retention_report.json`
    - `pointcloud_processing_report.json`
    - `scale_normalization_report.json`
    - `gaussian_splat/gaussian_splat_report.json`
    - `mesh/mesh_branch_report.json`
- Beta pipeline experiment data:
  - [run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv)
  - [stage_timings.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/system/stage_timings.csv)
  - [dust3r run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dust3r_data/run_overview/run_summary.csv)
  - [dust3r stage_timings.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dust3r_data/system/stage_timings.csv)
  - [RESULTS_SUMMARY.md](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md)
- Input-quality and frame-selection data:
  - [prefilter_summary.json](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/prefilter_summary.json)
  - [candidate_frame_scores.csv](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/candidate_frame_scores.csv)
  - [selected_frame_manifest.csv](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/selected_frame_manifest.csv)
  - [geometry_selection_summary.json](/D:/GitRepos/Thesis/experiments/geometry_test/results/irl_room_video_2_run_01/geometry_selection_summary.json)
- Fine-tuning run evidence:
  - [metrics.csv](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv)
  - [hparams.yaml](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/hparams.yaml)
  - [config_tree.log](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/config_tree.log)
  - [last.ckpt](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt)

## Why the selected technologies make sense

### Fast3R

- Chosen because it predicts dense multiview 3D structure from multiple input images in a learned forward-pass reconstruction workflow.
- Better suited than per-frame monocular depth for this project because it reasons across views instead of treating each frame independently.
- More practical than keeping a COLMAP-only path because the project needs dense, inspectable scene output quickly, while the archived sparse-COLMAP run produced only 1,230 sparse points and 24 reconstructed camera poses from 100 sampled frames in about 8 minutes.
- More appropriate than using a full NeRF or optimized 3DGS training workflow as the first app target because those workflows usually require iterative scene optimization after preprocessing and camera estimation.
- More appropriate than the older standalone DUSt3R experiment path because the project is centered on many-view indoor video reconstruction, and the local standalone DUSt3R route was slower and less stable in experiments.
- Local code anchors:
  - [fast3r_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/fast3r_pipeline.py)
  - [reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/reconstruction.py)

### PyTorch

- Used because Fast3R is implemented in PyTorch and the training/evaluation ecosystem is PyTorch Lightning/Hydra based.
- Enables GPU acceleration for inference and training.
- Supports checkpoint loading, mixed precision, and DeepSpeed-style training artifacts used by the Vast.ai fine-tuning run.

### PyQt6

- Chosen for a local desktop application rather than a web service.
- Fits the project because the user selects local videos, launches long-running reconstruction jobs, tracks progress, and opens local scene outputs.
- Avoids needing a browser-hosted or cloud-hosted UI for a GPU-heavy local prototype.
- Local code anchors:
  - [main.py](/D:/GitRepos/Thesis/src/main.py)
  - [main_window.py](/D:/GitRepos/Thesis/src/app/main_window.py)
  - [worker.py](/D:/GitRepos/Thesis/src/app/worker.py)

### OpenCV

- Used for video decoding, frame extraction, and visual frame-quality metrics.
- Supports blur/texture/brightness analysis before expensive model inference.
- Works well for scanning long videos and exporting selected frames at controlled resolutions.
- Local code anchor:
  - [video_processor.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/video_processor.py)
  - [frame_selection.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/frame_selection.py)

### Open3D

- Chosen for point-cloud IO, normal estimation, voxel downsampling, outlier cleanup, Poisson meshing, mesh decimation, and geometry metrics.
- It makes the postprocess part of the pipeline reproducible and scriptable without a separate DCC tool.
- Local code anchors:
  - [postprocess.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/postprocess.py)
  - [mesh_reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/mesh_reconstruction.py)

### NumPy and SciPy

- Used for dense point arrays, color arrays, camera-pose arrays, confidence filtering, nearest-neighbor scale estimation, and numerical reports.
- Supports saving intermediate arrays for debugging and thesis evidence.

### YAML and JSON

- YAML is used for user-editable runtime configuration:
  - [settings.yaml](/D:/GitRepos/Thesis/src/config/settings.yaml)
- JSON is used for scene metadata and reports because it is easy to inspect, compare, and convert into thesis tables.
- This was important because the thesis needs reproducible evidence, not only screenshots.

### Vast.ai

- Used for temporary remote GPU capacity for Fast3R fine-tuning.
- Appropriate because training is not the normal desktop-app workflow and would be expensive or impractical on the local machine.
- The thesis should present Vast.ai as an experimental training environment, not as the deployment environment of the app.

## Why the selected approaches make sense

### Geometry-aware frame selection

- Problem addressed:
  - raw video has thousands of frames
  - many frames are redundant or blurry
  - feeding every frame into reconstruction wastes time and memory
  - visual quality alone does not guarantee useful multiview geometry
- Selected approach:
  - scan the video using a stride
  - compute visual quality and redundancy metrics
  - build a visual shortlist
  - optionally run a low-resolution Fast3R probe
  - greedily select frames using geometry quality, diversity, overlap, temporal coverage, and pose validity
- Exact local evidence:
  - `irl_room_video_2.mp4` geometry-test scan:
    - fps: `60.10745716226093`
    - total frames: `10428`
    - original resolution: `848 x 478`
    - analysis resolution: `384 x 288`
    - scan stride: `15`
    - candidate frames: `696`
    - requested selected frames: `24`
    - selected frame count: `24`
    - probe pose valid count: `24`
    - probe pose valid ratio: `1.0`
  - source:
    - [geometry_selection_summary.json](/D:/GitRepos/Thesis/experiments/geometry_test/results/irl_room_video_2_run_01/geometry_selection_summary.json)
- Representative app run:
  - `scene_20260422_171717`
  - total video frames: `5373`
  - scan stride: `10`
  - candidate frames: `538`
  - shortlist size: `522`
  - selected frames: `180`
  - probe pose valid count: `180`
  - probe pose valid ratio: `1.0`
  - min selected gap: `10` frames
  - median selected gap: `30` frames

### Confidence and numeric filtering

- Problem addressed:
  - learned reconstruction can produce low-confidence points, invalid coordinates, or far-out outliers
  - downstream Gaussian and mesh branches need stable geometry
- Selected approach:
  - reject non-finite values
  - apply maximum absolute coordinate sanity filtering
  - apply confidence thresholds and confidence keep ratios
  - apply view-level confidence rejection
  - apply radius-percentile outlier filtering
  - optionally voxel downsample
- Representative run evidence:
  - `scene_20260422_171717`
  - numeric sanity input points: `26,264,029`
  - finite points: `26,264,029`
  - non-finite points dropped: `0`
  - large-coordinate points dropped: `0`
  - radius-percentile removed points: `78,793`
  - radius-percentile retention ratio: `0.9969999652376259`
  - radius-percentile threshold: `2.2449279833768503`
  - final points after cleanup/voxel: `8,146,338`

### Scale normalization

- Problem addressed:
  - learned reconstruction may remain in arbitrary scene units
  - indoor scene outputs need a usable meter-scale interpretation for viewing, meshing, and engine usage
- Selected approach:
  - first attempt metadata-based scale
  - if metadata lacks explicit meter/unit or paired baseline information, use geometric reference fallback
  - current fallback uses room-height reference:
    - reference name: `room_height`
    - real-world reference: `2.5` meters
    - axis: `z`
    - percentile interval: `2.0` to `98.0`
- Representative run evidence:
  - `scene_20260422_171717`
  - measured dimension in reconstruction units: `0.339406803548336`
  - scale formula: `real_world_dimension_meters / measured_dimension_in_reconstruction_units`
  - scale factor to meters: `7.365792240649551`
  - metadata scale failed because no explicit meter/unit field or paired baseline metadata was available
- Scale values across local app scenes:
  - `scene_20260421_224455`: `5.165823149853037`
  - `scene_20260422_003811`: `1.0`, failed scale reference, empty output after filtering
  - `scene_20260422_004756`: `3.946328578723078`
  - `scene_20260422_162029`: `4.041815749615849`
  - `scene_20260422_162950`: `12.0509716683903`
  - `scene_20260422_163940`: `6.777729205866679`
  - `scene_20260422_164436`: `7.433578903865953`
  - `scene_20260422_171717`: `7.365792240649551`

### Shared output branching

- Problem addressed:
  - Gaussian output and mesh output should describe the same scaled scene
  - running separate preprocessing paths would make outputs harder to compare and slower to generate
- Selected approach:
  - reconstruct and clean one scaled point cloud
  - branch after shared processing
  - Gaussian branch preserves dense point-based visual representation
  - mesh branch reduces/meshes the same geometry for object/engine-oriented use
- Representative run evidence:
  - `scene_20260422_171717`
  - final point cloud: `8,146,338` points
  - Gaussian output: `8,146,338` Gaussians
  - mesh branch:
    - input points: `8,146,338`
    - points after mesh voxel: `2,432,262`
    - points used for mesh: `500,000`
    - mesh point cap applied: `true`
    - Poisson depth: `9`
    - triangles before decimation: `2,872,061`
    - target triangles: `180,000`
    - final triangles: `179,965`
    - vertices: `85,591`
    - largest component ratio: `0.998655294084961`
    - watertight: `false`
    - edge manifold: `true`
    - vertex manifold: `false`
    - self intersecting: `true`
    - connected components: `32`

## Exact app output metrics available locally

| Scene | Video | Frames | Candidates | Shortlist | Probe valid | Raw points | Final points | Scale | Gaussians | Mesh triangles | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `scene_20260421_224455` | `legs_living.mp4` | 25 | 236 | 25 | 25 | 4,506,724 | 3,846,247 | 5.1658 | 3,846,247 | 249,923 | full Gaussian + mesh, 3 warnings |
| `scene_20260422_003811` | `no_legs_living.mp4` | 80 | 206 | 94 | 94 | 12,105,120 | 0 | 1.0 | none | none | scale reference invalid, empty point cloud after filtering |
| `scene_20260422_004756` | `no_legs_living.mp4` | 80 | 206 | 94 | 94 | 15,708,817 | 10,898,058 | 3.9463 | 10,898,058 | 249,926 | full Gaussian + mesh |
| `scene_20260422_162029` | `no_legs_living.mp4` | 180 | 309 | 303 | 180 | 19,889,399 | 5,942,945 | 4.0418 | 5,942,945 | none | mesh disabled or not present |
| `scene_20260422_162950` | `WhatsApp Video 2026-04-22 at 16.24.53.mp4` | 180 | 701 | 627 | 180 | 4,186,577 | 2,404,836 | 12.0510 | 2,404,836 | none | mesh disabled or not present |
| `scene_20260422_163940` | `irl_room_video_2.mp4` | 180 | 745 | 564 | 180 | 25,624,999 | 10,961,876 | 6.7777 | 10,961,876 | none | mesh disabled or not present |
| `scene_20260422_164436` | `WhatsApp Video 2026-04-22 at 16.39.55.mp4` | 150 | 597 | 550 | 150 | 17,850,216 | 6,338,134 | 7.4336 | 6,338,134 | none | mesh disabled or not present |
| `scene_20260422_171717` | `WhatsApp Video 2026-04-22 at 16.39.55.mp4` | 180 | 538 | 522 | 180 | 26,264,029 | 8,146,338 | 7.3658 | 8,146,338 | 179,965 | representative full branch run |

## Exact runtime and performance metrics available locally

### Historical early baseline: COLMAP and MiDaS

Source:

- [RESULTS_SUMMARY.md](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md)

Values:

| Method | Frames | Runtime | Output points / geometry | Strength | Limitation |
| --- | ---: | ---: | --- | --- | --- |
| COLMAP SfM | 100 | about 8 minutes | 1,230 sparse points, 24 recovered camera poses | clean sparse geometry and camera recovery | too sparse and slow for fast dense scene feedback |
| MiDaS-style monocular depth | 100 | about 35 seconds | 664,102 points before filtering, 637,362 after filtering, 2,184,537 mesh triangles | fast and dense | geometrically noisy, per-frame inconsistency, ambiguous scale |

Safe thesis wording:

- "The early COLMAP result showed that classical SfM gave cleaner geometric structure but was sparse and slow for the target feedback loop."
- "The early MiDaS-style result showed that dense per-frame depth was fast but produced noisy geometry and poor topology because it lacked multiview constraints."

### Standalone DUSt3R experiment route

Source:

- [dust3r run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dust3r_data/run_overview/run_summary.csv)
- [dust3r stage_timings.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dust3r_data/system/stage_timings.csv)

Values:

| Run | Frames | Status | Runtime | Raw points | Mesh vertices | Mesh triangles | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `20260404_150246` | 40 | failed | 21.63 s | 0 | 0 | 0 | reconstruction failed |
| `20260404_150830` | 40 | failed | 10.58 s | 0 | 0 | 0 | reconstruction failed |
| `20260404_150940` | 40 | success | 1604.15 s | 335,133 | 335,133 | 196,536 | very slow successful route |

Successful DUSt3R stage timings:

- frame extraction: `2.7082` s
- reconstruction: `1324.1504` s
- point-cloud processing: `12.1550` s
- mesh reconstruction: `261.3841` s
- save results: `3.7094` s

Safe thesis wording:

- "The standalone DUSt3R route was retained as experimental history but not chosen as the final application route because it produced failures and, in the successful 40-frame run, required about 26.7 minutes."

### Fast3R dual-output runs

Source:

- [run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv)
- [stage_timings.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/system/stage_timings.csv)

Selected values:

| Run | Frames | Runtime | Raw points | Processed/scaled points | Scale | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `20260422_064701` | 79 | 148.27 s | 6,306,527 | 6,306,527 | 3.9958 | success |
| `20260422_070036` | 120 | 223.68 s | 9,631,862 | 9,631,862 | 3.8049 | success |
| `20260422_090421` | 300 | 487.16 s | 14,918,694 | 14,918,694 | 7.2202 | success |
| `20260422_105125` | 180 | 235.06 s | 23,113,864 | 8,159,528 | 6.8617 | success |
| `20260422_111631` | 180 | 227.42 s | 30,196,929 | 30,166,735 | 6.7427 | failed |

Useful stage timing examples:

- `20260422_064701`, 79-frame success:
  - frame selection: `7.6549` s
  - inference: `108.3527` s
  - scale normalization: `1.4871` s
  - shared point-cloud processing: `2.7856` s
  - Gaussian branch: `27.6445` s
  - total runtime: `148.2671` s
  - GPU peak during inference: `13.6823` GB
- `20260422_070036`, 120-frame success:
  - frame selection: `7.4592` s
  - inference: `160.4089` s
  - scale normalization: `1.6433` s
  - total runtime: `223.6767` s
  - GPU peak during inference: `15.6586` GB

Failure evidence:

- HIP out-of-memory examples on a GPU reported with `15.92 GiB` total capacity.
- High-detail memory failure:
  - run `20260422_111631`
  - attempted 180-frame high-detail processing
  - raw points: `30,196,929`
  - processed/scaled points before failure: `30,166,735`
  - failure message: unable to allocate `230 MiB` for a `float64` array with shape `(30166735,)`

Safe thesis wording:

- "Fast3R made the project practical because it produced millions of dense points in minutes rather than requiring a long SfM/MVS or radiance-field optimization process."
- "The runtime evidence also shows why pruning and memory-aware settings are necessary: high-detail settings can fail after generating tens of millions of points."

## Fine-tuning metrics available locally

Source:

- [metrics.csv](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv)
- [last.ckpt](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt)

Run:

- run name: `arkitscenes_10hour_ga_head`
- dataset profile: ARKitScenes shortened run
- trainable scope: GA/global head only, mapped to `downstream_head`
- final remote checkpoint path recorded in logs:
  - `/workspace/logs/vast_finetune_arkitscenes/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt`
- local checkpoint directory:
  - [last.ckpt](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt)

Validation curve:

| Step | Trainer epoch | `val/loss` | Global confidence loss | Local confidence loss | Global 3D regression loss | Local 3D regression loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 999 | 0.1249 | 0.0838088393 | 0.0970245078 | 0.0705931783 | 0.1361721307 | 0.1228365973 |
| 1999 | 0.2499 | 0.0739670172 | 0.0915930420 | 0.0563409440 | 0.1370202154 | 0.1197275147 |
| 2999 | 0.3749 | 0.1046796069 | 0.1191352457 | 0.0902239755 | 0.1387095600 | 0.1226023734 |
| 3999 | 0.4999 | 0.0981354490 | 0.1098175794 | 0.0864533707 | 0.1369328052 | 0.1227595136 |
| 4999 | 0.6249 | 0.0918524191 | 0.1119885147 | 0.0717163309 | 0.1359119117 | 0.1210421398 |
| 5999 | 0.7499 | 0.0955662131 | 0.1102041379 | 0.0809282511 | 0.1379661262 | 0.1229711249 |
| 6999 | 0.8749 | 0.0677028298 | 0.0870897844 | 0.0483158827 | 0.1315391064 | 0.1160488799 |
| 7999 | 0.9999 | 0.0754956082 | 0.0924641937 | 0.0585269555 | 0.1340912580 | 0.1187858433 |

Computed values:

- first logged validation loss: `0.08380883932113647`
- final logged validation loss: `0.07549560815095901`
- final relative reduction from first validation point: `9.919277295230211%`
- best logged validation loss: `0.06770282983779907`
- best relative reduction from first validation point: `19.21755463242108%`
- checkpoint metadata:
  - `global_step = 7000`
  - `epoch = 0`
  - monitored metric: `val/loss`
  - `best_model_score = tensor(0.0677)`
  - best model path: `/workspace/logs/vast_finetune_arkitscenes/runs/arkitscenes_10hour_ga_head/checkpoints/epoch_000.ckpt`

Safe thesis wording:

- "During the completed shortened ARKitScenes GA-head fine-tuning run, validation loss decreased from 0.0838 to 0.0755, a 9.92% relative reduction between the first and final logged validation points. The best logged validation loss was 0.0677, corresponding to a 19.22% reduction relative to the first validation point."
- Do not write:
  - "the fine-tuned model improved by 10% over pretrained Fast3R"
  - unless paired pretrained and fine-tuned evaluation files are found or regenerated.

## Parameter values that should appear in thesis tables

### Main app preset values

Source:

- [settings.yaml](/D:/GitRepos/Thesis/src/config/settings.yaml)

Values:

- active preset: `high_quality_detail`
- target frames: `180`
- selection mode: `geometry_aware`
- visual shortlist target: `220`
- probe max frames: `180`
- scan stride: `10`
- min frame gap: `10`
- input analysis size: `384 x 288`
- output frame size: `1024 x 768`
- ORB features: `1000`
- Fast3R image size: `512`
- Fast3R dtype: `bfloat16`
- probe image size: `256`
- probe confidence threshold: `1.0`
- probe PnP iterations: `50`
- focal method: `first_view_from_global_head`
- model name: `jedyang97/Fast3R_ViT_Large_512`
- min confidence threshold: `1.05`
- confidence keep ratio: `0.85`
- view confidence p50 minimum: `1.2`
- view confidence p90 minimum: `1.45`
- weak texture retention: enabled
- max absolute coordinate: `100.0`
- outlier method: `radius_percentile`
- radius percentile: `99.7`
- voxel size: `0.001`
- scale reference: `room_height`
- scale reference real value: `2.5` meters
- scale axis: `z`
- scale percentile low/high: `2.0` / `98.0`
- Gaussian output enabled: true
- Gaussian nearest-neighbor scale multiplier: `0.35`
- Gaussian min/max scale: `0.00025` / `0.008`
- Gaussian density opacity: true
- Gaussian alpha min/max: `0.35` / `0.82`
- mesh output default in preset: false

### Representative mesh branch values

Source:

- [mesh_branch_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/mesh/mesh_branch_report.json)

Values:

- mesh method used in report: Poisson reconstruction
- input points: `8,146,338`
- points after mesh voxel: `2,432,262`
- points used for mesh: `500,000`
- mesh voxel size: `0.008`
- mesh max points: `500,000`
- Poisson depth: `9`
- Poisson scale: `1.1`
- Poisson trim quantile: `0.01`
- crop bounding-box margin: `0.03`
- connected components before cleanup: `5,493`
- components removed: `5,492`
- largest component triangles: `2,872,061`
- Laplacian iterations: `1`
- Laplacian lambda: `0.2`
- target triangles: `180,000`
- final triangles: `179,965`
- final vertices: `85,591`

### Representative Gaussian branch values

Source:

- [gaussian_splat_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/gaussian_splat/gaussian_splat_report.json)

Values:

- input points: `8,146,338`
- output Gaussians: `8,146,338`
- scale to meters already applied: `7.365792240649551`
- normal radius effective: `0.049324417114257814`
- normal max nearest neighbors: `64`
- density opacity: true
- alpha min/median/max: `0.3500014841556549` / `0.6547542810440063` / `0.8199999928474426`
- nearest-neighbor raw distance median: `0.0020567523315548897`
- nearest-neighbor raw distance p90: `0.004191207233816385`
- scale world median: `0.0007198632229119539`
- spherical harmonics initialization: `dc_only`
- nonzero higher-order SH coefficients: `0`
- rotations are identity: true

## Plot and figure plan from existing local data

### Figure 1: End-to-end pipeline diagram

- Data source:
  - code structure and existing diagram candidate [fast3r_geometry_aware_pipeline.svg](/D:/GitRepos/Thesis/thesis_docs/writing/fast3r_geometry_aware_pipeline.svg)
- Content:
  - video input
  - frame scanning
  - visual shortlist
  - geometry-aware probe
  - Fast3R reconstruction
  - filtering
  - scale normalization
  - Gaussian branch
  - mesh branch
  - scene metadata/viewer
- Thesis purpose:
  - explain architecture and contribution before metrics.

### Figure 2: Runtime versus frame count

- Data source:
  - [run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv)
- Plot:
  - x-axis: frames
  - y-axis: total runtime seconds
  - marker color: success/failure
  - marker size: raw point count or processed point count
- Values to highlight:
  - 79 frames -> `148.27` s
  - 120 frames -> `223.68` s
  - 180 frames -> `235.06` s for `presentation_180_geometry_retention`
  - 300 frames -> `487.16` s
- Thesis purpose:
  - show practical runtime scaling and memory pressure.

### Figure 3: Stage timing stacked bars

- Data source:
  - [stage_timings.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/system/stage_timings.csv)
- Plot:
  - each selected run as one stacked bar
  - segments:
    - frame selection
    - inference
    - scale normalization
    - shared point-cloud processing
    - Gaussian branch
    - mesh branch
- Values to highlight:
  - 79-frame run:
    - frame selection `7.65` s
    - inference `108.35` s
    - scale normalization `1.49` s
    - shared processing `2.79` s
    - Gaussian branch `27.64` s
  - 120-frame run:
    - frame selection `7.46` s
    - inference `160.41` s
    - scale normalization `1.64` s
- Thesis purpose:
  - show that inference dominates runtime and output branching is not the only cost.

### Figure 4: Method comparison table or grouped bar chart

- Data sources:
  - [RESULTS_SUMMARY.md](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md)
  - [dust3r run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dust3r_data/run_overview/run_summary.csv)
  - [Fast3R run_summary.csv](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv)
- Compare:
  - COLMAP SfM:
    - 100 frames
    - about `480` s
    - `1,230` sparse points
    - 24 recovered camera poses
  - MiDaS-style depth:
    - 100 frames
    - about `35` s
    - `664,102` points before filtering
    - noisy geometry
  - standalone DUSt3R:
    - 40 frames
    - `1604.15` s
    - `335,133` raw points
    - `196,536` mesh triangles
  - Fast3R dual-output:
    - 79 frames
    - `148.27` s
    - `6,306,527` raw points
    - Gaussian output generated
- Thesis purpose:
  - show why the project migrated toward Fast3R.
- Caveat:
  - this is an internal workflow comparison, not a controlled benchmark with identical output objectives.

### Figure 5: Point-retention waterfall for representative scene

- Data source:
  - [scene_metadata.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/scene_metadata.json)
  - [fast3r_point_retention_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/fast3r_point_retention_report.json)
  - [pointcloud_processing_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/pointcloud_processing_report.json)
- Plot:
  - bar/waterfall:
    - predicted dense points, if using point-retention report: `35,389,440`
    - extracted raw points: `26,264,029`
    - sane points: `26,264,029`
    - filtered points: `26,185,236`
    - final points: `8,146,338`
- Thesis purpose:
  - explain why the pipeline needs filtering and downsampling even after successful inference.

### Figure 6: Selected frame timeline

- Data source:
  - `selected_frame_indices` in [scene_metadata.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/scene_metadata.json)
  - [geometry_selection_summary.json](/D:/GitRepos/Thesis/experiments/geometry_test/results/irl_room_video_2_run_01/geometry_selection_summary.json)
- Plot:
  - x-axis: video frame index
  - y-axis: selected/not selected or selected score
  - optionally overlay selected frame score, visual score, geometry score
- Thesis purpose:
  - show temporal coverage and explain why selected frames are sparse but representative.

### Figure 7: Input-quality score distribution

- Data source:
  - [candidate_frame_scores.csv](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/candidate_frame_scores.csv)
  - [selected_frame_manifest.csv](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/selected_frame_manifest.csv)
  - [prefilter_summary.json](/D:/GitRepos/Thesis/experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/prefilter_summary.json)
- Values:
  - candidate frames: `696`
  - selected/exported frames: `80`
  - mean selected score: `0.9559575828537344`
  - median selected score: `0.9557598873972893`
- Plot:
  - histogram of candidate scores
  - selected frames highlighted
  - optional top selected source indices as annotations
- Thesis purpose:
  - justify preprocessing and input curation.

### Figure 8: Scale factor per scene

- Data source:
  - all [scene_metadata.json](/D:/GitRepos/Thesis/src/outputs) files
- Plot:
  - scene id on x-axis
  - scale factor on y-axis
  - failed/identity scale highlighted
- Values:
  - range in successful local scenes: about `3.9463` to `12.0510`
  - identity/failure case: `scene_20260422_003811`, scale `1.0`, empty output after filtering
- Thesis purpose:
  - show why scale normalization is a real part of the pipeline, not a cosmetic step.

### Figure 9: Mesh simplification and quality report

- Data source:
  - [mesh_branch_report.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/mesh/mesh_branch_report.json)
- Plot/table:
  - input points: `8,146,338`
  - mesh voxel points: `2,432,262`
  - points used for mesh: `500,000`
  - triangles before decimation: `2,872,061`
  - final triangles: `179,965`
  - components before cleanup: `5,493`
  - components removed: `5,492`
  - largest component ratio: `0.998655294084961`
- Thesis purpose:
  - show why mesh generation is useful but still limited.

### Figure 10: Fine-tuning validation loss curve

- Data source:
  - [metrics.csv](/D:/GitRepos/Thesis/experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv)
- Plot:
  - x-axis: step
  - y-axis: validation loss
  - mark first, final, and best validation points
- Values:
  - first `0.0838088393`
  - final `0.0754956082`
  - best `0.0677028298`
  - first-to-final relative reduction `9.92%`
  - first-to-best relative reduction `19.22%`
- Thesis purpose:
  - show that the GA-head training run completed and produced measurable validation-loss movement.
- Caveat:
  - this curve is within one fine-tuning run, not a paired baseline-vs-finetuned benchmark.

### Figure 11: Visual output examples

- Data source:
  - generated outputs under [src/outputs](/D:/GitRepos/Thesis/src/outputs)
  - representative complete run [scene_20260422_171717](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717)
- Images to capture or render:
  - selected input frames montage
  - point cloud screenshot
  - Gaussian-splat viewer screenshot
  - mesh screenshot
  - failure/partial-output screenshot from `scene_20260422_003811` or `presentation_180_high_detail`
- Thesis purpose:
  - make the application tangible and not only metric-based.
- Important restriction:
  - do not insert old/pretrained visual output screenshots as final model-output figures.
  - for the final thesis, real visual model outputs should be shown only after they are regenerated with the new/fine-tuned Fast3R weights.
  - until that regeneration exists locally, keep this as a planned capture category rather than a guaranteed thesis figure.

## Exact thesis figure/table markers and reproducible scripts

Use the following markers exactly in the final thesis draft. Every item below is backed by local data and has a script in `helper/script_for_plotting` that either generates a PNG plot or writes a Markdown table/diagram into `helper/script_for_plotting/generated`.

| Marker to insert | Thesis location | Local data behind it | Script |
| --- | --- | --- | --- |
| `[FIGURA 1.1 - Contributiile principale ale lucrarii]` | Chapter 1, after the contribution list | Project architecture and contribution summary from local code/docs; conceptual diagram, not numeric data | `helper/script_for_plotting/figura_1_1_contributiile_principale_ale_lucrarii.py` |
| `[TABELUL 2.1 - Comparatia principalelor solutii analizate]` | Chapter 2, methods/related systems comparison | `experiments/beta_pipeline_testing/docs/archive/RESULTS_SUMMARY.md`, `experiments/beta_pipeline_testing/dust3r_data/run_overview/run_summary.csv`, `experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv` | `helper/script_for_plotting/tabelul_2_1_comparatia_principalelor_solutii_analizate.py` |
| `[FIGURA 3.1 - Selectia cadrelor pe axa temporala]` | Chapter 3, frame selection and input preparation | `src/outputs/scene_20260422_171717/scene_metadata.json`, especially `selected_frame_indices`, `total_frames`, `selected_frame_count` | `helper/script_for_plotting/figura_3_1_selectia_cadrelor_pe_axa_temporala.py` |
| `[FIGURA 3.2 - Distributia scorurilor de prefiltrare]` | Chapter 3, input quality filtering | `experiments/input_quality_lab/results/20260419_205738_fast3r_prefilter/candidate_frame_scores.csv` and `selected_frame_manifest.csv` | `helper/script_for_plotting/figura_3_2_distributia_scorurilor_de_prefiltrare.py` |
| `[TABELUL 5.1 - Parametrii principali ai pipeline-ului]` | Chapter 5, application architecture and configuration | `src/config/settings.yaml` active Fast3R parameters: frames, selection mode, image size, dtype, confidence filters, scale, Gaussian/mesh flags | `helper/script_for_plotting/tabelul_5_1_parametrii_principali_ai_pipeline_ului.py` |
| `[FIGURA 6.1 - Runtime in functie de numarul de cadre]` | Chapter 6, application performance | `experiments/beta_pipeline_testing/dual_output_fast3r_data/run_overview/run_summary.csv` | `helper/script_for_plotting/figura_6_1_runtime_in_functie_de_numarul_de_cadre.py` |
| `[FIGURA 6.2 - Timpii pe etape pentru rulari reprezentative]` | Chapter 6, time-to-inspectable-scene analysis | `experiments/beta_pipeline_testing/dual_output_fast3r_data/system/stage_timings.csv` | `helper/script_for_plotting/figura_6_2_timpii_pe_etape_pentru_rulari_reprezentative.py` |
| `[TABELUL 6.1 - Sumarul scenelor reconstruite local]` | Chapter 6, reconstructed-scene evidence | All `src/outputs/scene_*/scene_metadata.json` files available locally | `helper/script_for_plotting/tabelul_6_1_sumarul_scenelor_reconstruite_local.py` |
| `[FIGURA 6.3 - Retentia punctelor pentru scena reprezentativa]` | Chapter 6, point filtering and output scale | `src/outputs/scene_20260422_171717/fast3r_point_retention_report.json` and `scene_metadata.json` | `helper/script_for_plotting/figura_6_3_retentia_punctelor_pentru_scena_reprezentativa.py` |
| `[TABELUL 6.2 - Calitatea meshului pentru scena reprezentativa]` | Chapter 6, mesh branch evaluation | `src/outputs/scene_20260422_171717/mesh/mesh_branch_report.json` | `helper/script_for_plotting/tabelul_6_2_calitatea_meshului_pentru_scena_reprezentativa.py` |
| `[FIGURA 6.4 - Curba de validare pentru fine-tuning GA-head]` | Chapter 6, model adaptation experiment | `experiments/vast_training/runs/runs/arkitscenes_10hour_ga_head/csv/version_0/metrics.csv`, validation rows only | `helper/script_for_plotting/figura_6_4_curba_de_validare_pentru_fine_tuning_ga_head.py` |
| `[TABELUL 6.3 - Valorile de validare pentru fine-tuning GA-head]` | Chapter 6, model adaptation numeric evidence | Same `metrics.csv`; table of validation loss and component losses | `helper/script_for_plotting/tabelul_6_3_valorile_de_validare_pentru_fine_tuning_ga_head.py` |

Important wording rule for visual examples:

- plots/tables based on local metadata and logs may be used as evidence of application performance and pipeline behavior;
- screenshots/renders of reconstructed scenes should only be inserted into the final thesis if they come from Fast3R with the new/fine-tuned weights;
- no current local screenshot is promoted here as a guaranteed final thesis model-output figure under that restriction.

## Manual figure registry for thesis visuals not generated by scripts

The scripted registry above is intentionally conservative because it only includes plots/tables that can be generated from local structured files today. The final thesis should also contain manually produced visuals. These are not Python-script artifacts, but they should still be referenced with exact markers so the thesis generator knows where they belong.

| Marker to insert | Thesis location | What to produce manually | Safe status |
| --- | --- | --- | --- |
| `[FIGURA 1.2 - Fluxul general al lucrarii de la video la scena inspectabila]` | Chapter 1, after problem statement or contribution overview | Manual diagram showing video input -> frame selection -> Fast3R -> shared geometry processing -> Gaussian/mesh outputs -> viewer/evaluation | Safe to create now as a conceptual diagram |
| `[FIGURA 2.1 - Pozitionarea Fast3R fata de reconstructia iterativa clasica]` | Chapter 2, after reconstruction-method discussion | Manual comparison diagram showing Fast3R forward-pass multiview inference versus COLMAP/MVS/training-heavy workflows | Safe to create now as a conceptual diagram |
| `[FIGURA 5.1 - Arhitectura aplicatiei desktop]` | Chapter 5, application architecture | UML/component SVG: PyQt UI, application services, pipeline layer, infrastructure/files | `thesis_docs/document/figures/application/figura_5_1_arhitectura_aplicatiei_desktop.svg` |
| `[FIGURA 5.2 - Fluxul UI scene-first al aplicatiei]` | Chapter 5, implementation workflow | SVG workflow: New Scene, Analyze Video, Run Reconstruction, Open Output, Scene Library, Reconstruction Properties, embedded viewer | `thesis_docs/document/figures/application/figura_5_2_fluxul_ui_scene_first.svg` |
| `[FIGURA 5.3 - Structura outputului pentru o scena reconstruita]` | Chapter 5, output organization | SVG artifact diagram: metadata, reports, point cloud, frames, Gaussian folder, optional mesh folder | `thesis_docs/document/figures/application/figura_5_3_structura_outputului_scene.svg` |
| `[FIGURA 5.4 - Interfata principala a aplicatiei desktop]` | Chapter 5, UI evidence | Screenshot of the running desktop app with Scene Library, central workspace, Reconstruction Properties, and bottom progress tracker | Safe to capture now; UI screenshot, not model output |
| `[FIGURA 5.5 - Dialogul New Scene si selectia profilului de greutati]` | Chapter 5, scene workflow | Screenshot of New Scene dialog with scene name, source video, description, pipeline, Default/Fine-Tuned Fast3R profile, preset, tags | Safe to capture now; UI screenshot |
| `[FIGURA 5.6 - Scene Library si panoul Reconstruction Properties]` | Chapter 5, scene management and configuration | Screenshot focused on scene selection, adaptive scene table, input video, weights profile, preset, status, actions | Safe to capture now; UI screenshot |
| `[FIGURA 5.7 - Viewerul Viser incorporat pentru scena reconstruita]` | Chapter 5, output inspection | Screenshot of embedded Viser viewer inside the PyQt central workspace; if 3D output is visible, use only a scene regenerated with fine-tuned/new weights | Conditional: UI shell safe now; visible model output must follow new-weight rule |
| `[FIGURA 6.5 - Montaj cu cadrele selectate pentru scena evaluata]` | Chapter 6, before output/evaluation figures | Montage of 8-12 selected input frames from the representative/new-weight scene | Safe if frames are input frames; not a model-output claim |
| `[FIGURA 6.6 - Output Gaussian Fast3R cu greutatile noi]` | Chapter 6, qualitative results | Screenshot/render from the Gaussian viewer for the scene regenerated with the new/fine-tuned Fast3R weights | Blocked until the scene is regenerated with new weights |
| `[FIGURA 6.7 - Output mesh Fast3R cu greutatile noi]` | Chapter 6, qualitative results | Screenshot/render of the mesh branch for the same scene regenerated with the new/fine-tuned Fast3R weights | Blocked until the scene is regenerated with new weights |
| `[FIGURA 6.8 - Comparatie vizuala intre output Gaussian si output mesh]` | Chapter 6, qualitative results | Two-panel figure comparing Gaussian and mesh output for the same regenerated new-weight scene | Blocked until both outputs are regenerated with new weights |
| `[FIGURA 6.9 - Exemplu de limitare: memorie sau output partial]` | Chapter 6, limitations/failure analysis | Screenshot of log/error, memory failure, or partial output that illustrates high-detail/OOM limits | Log screenshot is safe now; visual model output must be new-weight only |
| `[FIGURA 7.1 - Sinteza limitarilor si directiilor viitoare]` | Chapter 7, conclusion | Manual diagram summarizing limitations and future work: scale, memory, mesh topology, controlled benchmark, visual new-weight outputs | Safe to create now as a conceptual diagram |

Manual visual priority:

1. Highest priority:
   - `[FIGURA 5.4 - Interfata principala a aplicatiei desktop]`
   - `[FIGURA 5.5 - Dialogul New Scene si selectia profilului de greutati]`
   - `[FIGURA 5.6 - Scene Library si panoul Reconstruction Properties]`
   - `[FIGURA 5.7 - Viewerul Viser incorporat pentru scena reconstruita]`
   - `[FIGURA 6.6 - Output Gaussian Fast3R cu greutatile noi]`
   - `[FIGURA 6.8 - Comparatie vizuala intre output Gaussian si output mesh]`
2. Second priority:
   - `[FIGURA 5.1 - Arhitectura aplicatiei desktop]`
   - `[FIGURA 5.2 - Fluxul UI scene-first al aplicatiei]`
   - `[FIGURA 5.3 - Structura outputului pentru o scena reconstruita]`
   - `[FIGURA 6.5 - Montaj cu cadrele selectate pentru scena evaluata]`
   - `[FIGURA 6.9 - Exemplu de limitare: memorie sau output partial]`
3. Nice to have:
   - `[FIGURA 1.2 - Fluxul general al lucrarii de la video la scena inspectabila]`
   - `[FIGURA 2.1 - Pozitionarea Fast3R fata de reconstructia iterativa clasica]`
   - `[FIGURA 7.1 - Sinteza limitarilor si directiilor viitoare]`

## Draft placeholder registry extracted from `experiments/gpt` IDE text

The local file `experiments/gpt` contains 26 placeholder markers. Keep these as useful thesis planning items. They are not all guaranteed artifacts. Some are internal/local and can be produced from repository data; some are conceptual diagrams; some are external/context figures that must be redrawn by us or cited properly. Do not copy third-party paper figures directly unless license/permission and citation are handled.

Numbering warning:

- The draft uses its own numbering, for example `[FIGURA 1.3 - ...]` and `[FIGURA 2.1 - ...]`.
- The existing registry above already has other markers with overlapping numbers.
- Before final thesis generation, pick one consistent numbering scheme and update all cross-references.

| Draft marker | Category | How to use / produce |
| --- | --- | --- |
| `[FIGURA 1.3 – Contribuțiile principale ale lucrării]` | Conceptual/manual | Same content family as `[FIGURA 1.1 - Contributiile principale ale lucrarii]`; create a clean contribution diagram with frame selection, Fast3R, scale normalization, Gaussian, mesh, performance, GA-head fine-tuning |
| `[FIGURA 1.4 – Metodologia generală de dezvoltare a sistemului]` | Conceptual/manual | Development-methodology diagram: solution analysis -> Fast3R choice -> frame selection -> postprocessing -> scale normalization -> dual output -> fine-tuning |
| `[FIGURA 2.1 – Fluxul general al tehnologiilor utilizate în cadrul sistemului]` | Conceptual/manual | System technology flow: Python/PyQt6 app, OpenCV/frame selection, Fast3R/PyTorch, Open3D, Gaussian output, mesh output |
| `[FIGURA 2.2 – Evoluția abordărilor de reconstrucție 3D]` | External/context/manual | Recreate as our own timeline/overview with citations: SfM/MVS, monocular depth, learned multiview models, Gaussian Splatting; do not paste external figures |
| `[FIGURA 2.3 – Ramificarea pipeline-ului către reprezentările Gaussian și Mesh]` | Conceptual/manual | Diagram showing shared expensive stages once, then branching to Gaussian and mesh; strong performance contribution |
| `[FIGURA 2.4 – Exemple de cadre cu niveluri diferite de calitate]` | Manual/local images | Use our own input-video frames: sharp, blurred, over/underexposed, low texture; not a model-output figure |
| `[FIGURA 2.5 – Procesul de selecție geometry-aware]` | Conceptual/manual | Diagram: visual quality scoring -> redundancy reduction -> geometry probe -> final selected frames |
| `[FIGURA 2.6 – Exemplu de nor de puncte generat de sistem]` | Model output visual | Only include if generated from Fast3R with new/fine-tuned weights |
| `[FIGURA 2.7 – Efectul normalizării scalei]` | Local/manual or model output | Safe as a diagram/table from scale metadata; if using 3D screenshots, only new-weight outputs |
| `[FIGURA 2.8 – Exemplu de reprezentare Gaussian Splatting]` | Model output visual | Only include if rendered from new/fine-tuned Fast3R output |
| `[FIGURA 2.9 – Generarea unui model Mesh prin reconstrucție Poisson]` | Conceptual/manual | Create our own schematic: processed point cloud -> normal estimation -> Poisson reconstruction -> cleanup/decimation |
| `[FIGURA 3.1 – Fluxul datelor în cadrul sistemului]` | Conceptual/manual | Data-flow diagram from raw video to selected frames, inference, metadata, Gaussian, mesh, viewer |
| `[FIGURA 3.2 – Exemple de cadre eliminate în etapa de preprocesare]` | Manual/local images | Use rejected/low-quality candidate frames from local videos or prefilter outputs |
| `[FIGURA 3.3 – Procesul de selecție geometry-aware]` | Conceptual/manual | Duplicate topic with `[FIGURA 2.5 – Procesul de selecție geometry-aware]`; keep only one in final numbering unless chapter structure needs both |
| `[FIGURA 3.4 – Efectele filtrării și normalizării]` | Local/manual or model output | Can be represented by point-retention/scale metadata; visual 3D before/after requires new-weight output |
| `[FIGURA 3.5 – Fluxul de pregătire a datelor pentru fine-tuning]` | Conceptual/manual | ARKitScenes preparation diagram: raw scenes -> metadata conversion -> train/val split -> Vast.ai training run |
| `[FIGURA 4.1 – Poziția modelului Fast3R în cadrul pipeline-ului]` | Conceptual/manual | Diagram showing Fast3R as central inference block within broader app pipeline |
| `[FIGURA 4.2 – Fluxul de integrare al modelului Fast3R]` | Conceptual/manual | Detailed model-integration flow: selected images -> preprocessing -> Fast3R runner -> confidence outputs -> postprocessing |
| `[FIGURA 4.3 – Strategia de adaptare a modelului]` | Conceptual/manual | Head-only training schematic: frozen model body/backbone, trainable GA head, ARKitScenes supervision |
| `[TABELUL 2.2 – Justificarea tehnologiilor utilizate]` | Manual/local table | Technology, role, reason chosen, limitation: Fast3R, PyTorch, PyQt6, OpenCV, Open3D, Gaussian, mesh, YAML/JSON, Vast.ai |
| `[TABELUL 2.3 – Impactul strategiilor de selecție a cadrelor]` | Data table if comparable runs exist | Use only if local runs support comparable strategies; otherwise convert to qualitative discussion or future-work table |
| `[TABELUL 2.4 – Comparația dintre reprezentările Point Cloud, Gaussian Splatting și Mesh]` | Conceptual/manual table | Compare representation, strengths, limitations, best use, relation to this project |
| `[TABELUL 3.1 – Statistici ale procesului de selecție]` | Local data table | Derive from `scene_metadata.json`, `candidate_frame_scores.csv`, `selected_frame_manifest.csv`: total frames, candidates, shortlist, selected frames, selected-score stats |
| `[TABELUL 4.1 – Parametrii principali utilizați în procesul de inferență]` | Local data table | Similar to generated `[TABELUL 5.1 - Parametrii principali ai pipeline-ului]`; can be chapter-renumbered |
| `[TABELUL 4.2 – Configurația utilizată pentru fine-tuning]` | Local data table | Use Vast/ARKitScenes GA-head configuration and training logs; avoid presenting it as central contribution |

Draft placeholders that should be merged or renamed:

- `[FIGURA 1.3 – Contribuțiile principale ale lucrării]` and generated `[FIGURA 1.1 - Contributiile principale ale lucrarii]` describe the same thesis idea.
- `[FIGURA 2.5 – Procesul de selecție geometry-aware]` and `[FIGURA 3.3 – Procesul de selecție geometry-aware]` are likely duplicates.
- `[TABELUL 4.1 – Parametrii principali utilizați în procesul de inferență]` overlaps with generated `[TABELUL 5.1 - Parametrii principali ai pipeline-ului]`.
- Any visual model output placeholder must respect the rule: only Fast3R with new/fine-tuned weights is allowed for final qualitative screenshots.

## Strong claims and safe wording

- Strong safe claim:
  - "The system achieved a complete local video-to-scene pipeline around Fast3R, including frame selection, scale normalization, Gaussian-splat initialization, mesh export, metadata storage, and desktop interaction."
- Strong safe claim:
  - "A representative completed scene used 180 selected frames and produced 8.15 million final points/Gaussians after processing."
- Strong safe claim:
  - "The representative mesh branch reduced millions of input points into a 179,965-triangle mesh, but the resulting mesh was not watertight and still contained topology issues."
- Strong safe claim:
  - "The internal Fast3R runtime evidence supports the project focus on time-to-inspectable-scene, with selected runs completing in minutes while generating millions of points."
- Strong safe claim:
  - "The shortened GA-head fine-tuning run completed and reduced logged validation loss from 0.0838 to 0.0755 between first and final validation points."
- Do not claim:
  - "real-time reconstruction"
  - "metric accuracy proven against ground truth"
  - "fully optimized Gaussian splatting"
  - "10% improvement over pretrained baseline"
  - "formal benchmark against COLMAP dense MVS, Nerfstudio, INRIA 3DGS, or Postshot"

## Recommended thesis table list

- Table: technology choices and justification
  - Fast3R, PyTorch, PyQt6, OpenCV, Open3D, YAML/JSON, Vast.ai
- Table: method evolution
  - MiDaS-style depth -> COLMAP/SfM -> standalone DUSt3R -> Fast3R pipeline
- Table: representative app scene outputs
  - scene id, frames, raw points, final points, scale, Gaussians, mesh triangles
- Table: runtime summary
  - run id, frames, runtime, raw points, processed points, status
- Table: stage timing summary
  - frame selection, inference, scale normalization, processing, Gaussian branch
- Table: mesh branch quality
  - points used, triangles before/after decimation, watertight/manifold/self-intersection flags
- Table: fine-tuning validation loss
  - step, validation loss, component losses, relative reduction
- Table: limitations and future work
  - missing ground truth, no full 3DGS optimization, topology limitations, memory pressure, no paired fine-tuned baseline comparison yet

## Recommended chapter allocation for 45-page thesis

- Introduction: 5 to 6 pages
  - problem, motivation, speed gap, indoor-video difficulty, contribution list
- Background: 8 to 10 pages
  - multiview reconstruction, Fast3R lineage, Gaussian splatting, meshing, scale ambiguity, frame selection
- Dataset and preprocessing: 5 to 6 pages
  - video input characteristics, frame scoring, geometry-aware selection, ARKitScenes training prep
- Model and training: 5 to 6 pages
  - Fast3R use, GA-head-only fine-tuning rationale, Vast.ai setup, validation-loss curve, caveats
- Application design: 7 to 8 pages
  - PyQt UI, pipeline architecture, scene metadata, output branching, configuration
- Evaluation and results: 8 to 10 pages
  - runtime tables, representative scenes, point retention, scale, Gaussian/mesh outputs, failure cases, fine-tuning metrics
- Conclusions and future work: 3 to 4 pages
  - achieved pipeline, strongest points, limitations, next steps

# 7. Conclusions and Future Work

## Summary of system and model performance

- Conclusion marker:
  - project achieved a working end-to-end Fast3R desktop reconstruction pipeline
  - strongest engineering additions are frame curation, confidence-based retention, scale normalization, and shared-output branching
- Conclusion marker:
  - performance is a primary design goal and one of the strongest application-level differentiators
  - the intended user-facing metric is time-to-inspectable-scene
  - current evidence supports the direction through local Fast3R timings and a historical sparse-COLMAP baseline, while a controlled cross-tool benchmark remains future work
- Conclusion marker:
  - geometry quality improved through more selective preprocessing and postprocess filtering, but scale still often depends on geometric assumptions rather than intrinsic metadata
- Conclusion marker:
  - Gaussian output is strong as a dense visual representation
  - mesh output is usable but still shows topology limitations on difficult indoor scenes
- Conclusion marker:
  - memory/performance tradeoffs are central; high-detail settings can become impractical without stronger hardware or tighter pruning
- Evidence anchor:
  - [src/EXPERIMENTS_AUDIT.md](/D:/GitRepos/Thesis/src/EXPERIMENTS_AUDIT.md)
  - [scene_metadata.json](/D:/GitRepos/Thesis/src/outputs/scene_20260422_171717/scene_metadata.json)
- Missing:
  - one final chosen "best configuration" statement justified by both quality and runtime

## Future improvements

- Fine-tuning marker:
  - one shortened GA/global-head indoor fine-tuning run completed and has local validation-loss evidence
  - future work should run a controlled pretrained-versus-fine-tuned comparison and measure whether alignment, walls, corners, and low-texture surfaces improve in final app outputs
  - current training/evaluation code located in [experiments/vast_training](/D:/GitRepos/Thesis/experiments/vast_training)
- Data marker:
  - preserve the exact ARKitScenes subset metadata used for the completed run and, if time allows, add a second public indoor validation set
- Geometry marker:
  - compare Poisson against Screened Poisson, BPA, or advancing-front style meshing on the same cleaned clouds
- Selection marker:
  - complete formal ablation between uniform, coverage-aware, and geometry-aware frame selection
- Scale marker:
  - add richer absolute-scale recovery using better scene priors or capture metadata
- Output marker:
  - train or refine Gaussian representation further if true view synthesis quality is a target
- App marker:
  - richer viewer, batch processing, export presets, mesh quality modes
- Semantics marker:
  - semantic segmentation
  - object separation
  - physics-aware scene preparation
- Missing:
  - explicit priority ordering for future work based on thesis scope limits

# 8. Bibliography

- Candidate primary citations to collect into BibTeX:
  - Fast3R paper / repo
  - DUSt3R paper
  - Spann3R paper
  - Poisson Surface Reconstruction, 2006
  - Screened Poisson Surface Reconstruction, 2013
  - Ball Pivoting Algorithm, 1999
  - Marching Cubes, 1987
  - Volumetric fusion / TSDF, 1996
  - Open3D documentation
- Local bibliography support file:
  - [mesh_reconstruction_from_ply_sources_2026-04-06.md](/D:/GitRepos/Thesis/thesis_docs/research_notes/mesh_reconstruction_from_ply/mesh_reconstruction_from_ply_sources_2026-04-06.md)
- Missing:
  - actual `.bib` file
  - exact Fast3R / DUSt3R / Spann3R reference details collected in thesis format

# Appendices (Optional)

## Installation steps

- App-side run marker:
  - documented in [src/README.md](/D:/GitRepos/Thesis/src/README.md)
  - run via `python -m src.main` or `./run_app.sh`
- Fine-tuning/Vast marker:
  - documented in [experiments/vast_training/README.md](/D:/GitRepos/Thesis/experiments/vast_training/README.md)
- Missing:
  - one cleaned thesis appendix with only the final supported setup steps

## Code excerpts

- Good excerpt candidates:
  - geometry-aware frame scoring and greedy selection from [frame_selection.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/frame_selection.py)
  - scale normalization logic from [scale_normalization.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/scale_normalization.py)
  - output branching from [fast3r_pipeline.py](/D:/GitRepos/Thesis/src/pipeline/fast3r_pipeline.py)
  - Poisson cleanup sequence from [mesh_reconstruction.py](/D:/GitRepos/Thesis/src/pipeline/fast3r/mesh_reconstruction.py)
- Missing:
  - decide whether thesis wants pseudocode or short real-code excerpts

## Screenshots or demo links

- Figure / media candidates available in repo:
  - [fast3r_geometry_aware_pipeline.svg](/D:/GitRepos/Thesis/thesis_docs/writing/fast3r_geometry_aware_pipeline.svg)
  - generated outputs under [src/outputs](/D:/GitRepos/Thesis/src/outputs)
  - experimental outputs under [experiments/beta_pipeline_testing/outputs](/D:/GitRepos/Thesis/experiments/beta_pipeline_testing/outputs)
- Missing:
  - curated app screenshots
  - curated viewer screenshots
  - thesis-ready before/after reconstruction images
  - demo link or video clip list

## Open Gaps Checklist

- `my_work.docx` missing; author-style calibration still blocked
- no finalized thesis title
- no final BibTeX database yet
- no complete ground-truth geometric benchmark section yet
- no paired pretrained-versus-fine-tuned comparison bundle found locally for the completed GA-head run
- no curated screenshot pack yet
- no single consolidated parameter table yet
- no final ablation table for selector variants yet
- no controlled cross-tool runtime benchmark yet for COLMAP dense MVS, Nerfstudio Splatfacto, INRIA 3DGS, Postshot, and this application

## Repository navigation map for future agents

This section exists so a future agent can orient quickly in the repository without re-discovering the layout from scratch.

### Full inventory script

There is now a root-level inventory script that starts from the repository root and prints every discovered directory and file, while excluding only environment and cache noise:

- script path:
  - `scripts/print_repo_inventory.py`
- intended usage from repository root:
  - `python scripts/print_repo_inventory.py`

Exclusions used by the script:

- `.git`
- `.venv`
- `.pip_tmp`
- `__pycache__`
- `.pytest_cache`
- `.mypy_cache`
- `.ruff_cache`
- `node_modules`

This script was written and test-run successfully from the root working directory. It prints the repository map in a stable `[DIR]` / `[FILE]` format so another agent can regenerate a complete inventory whenever needed.

### Practical high-level repository map

The project is large, so the most useful navigation aid is a hierarchy of the important roots and what they contain.

#### Root level

- `agent/`
  - persistent Codex work log and recovery notes
- `config/`
  - project-wide configuration
- `data/`
  - local models and processed data support folders
- `experiments/`
  - archived experiments, comparisons, plots, and Vast training work
- `fast3r/`
  - external model codebase used by the project
- `scripts/`
  - utility scripts, including the repository inventory printer
- `src/`
  - main desktop application and reconstruction pipeline
- `thesis_docs/`
  - thesis writing material, templates, research notes, figures, and generated draft content

Representative root files worth knowing immediately:

- `README.md`
- `requirements.txt`
- `script.txt`
- `PIPELINE_EXPLICATIE_SRC.txt`
- `RECONSTRUCTION_IMPROVEMENTS_2026_04_22.md`

#### `src/` – main application

- `src/main.py`
  - application entry point
- `src/app/`
  - UI, worker, scene management integration
- `src/config/`
  - runtime settings for the app pipeline
- `src/models/`
  - data models such as scene result structures
- `src/pipeline/`
  - pipeline factory, base pipeline, Fast3R orchestration
- `src/pipeline/fast3r/`
  - reconstruction modules:
    - video processing
    - frame selection
    - Fast3R reconstruction
    - scale normalization
    - postprocess
    - Gaussian export
    - mesh reconstruction
- `src/viewer/`
  - local scene viewer
- `src/outputs/`
  - saved scene runs and their reports
- `src/data/`
  - local indexes and app-side metadata

#### `experiments/` – experimental evidence

- `experiments/beta_pipeline_testing/`
  - archived application-side experiments and summaries
- `experiments/geometry_test/`
  - geometry-aware frame selection work
- `experiments/input_quality_lab/`
  - input filtering and diagnostics
- `experiments/gaussian_splatting_methods/`
  - Gaussian output experiments
- `experiments/mesh_reconstrction_methods/`
  - mesh cleanup and reconstruction comparisons
- `experiments/raw_inference_output/`
  - inspection of raw Fast3R outputs
- `experiments/vast_training/`
  - fine-tuning, evaluation, conversion, bundling, and callback code
- `experiments/vast_training/configs/`
  - JSON experiment profiles used for training and evaluation
- `experiments/vast_training/runs/`
  - downloaded remote run artifacts

#### `thesis_docs/` – thesis writing workspace

- `thesis_docs/document/`
  - thesis draft, template, project documentation reference, knowledge files
- `thesis_docs/document/model_finetuning/`
  - fine-tuning writing helpers and templates
- `thesis_docs/document/figure_data_plan/`
  - figure planning and plotting helpers
- `thesis_docs/document/application_imporvements/`
  - architecture diagrams and app-design notes
- `thesis_docs/research_notes/`
  - technical note collections used to support thesis claims
- `thesis_docs/writing/`
  - visual aids and intermediate thesis-writing assets

Most important thesis files at this moment:

- `thesis_docs/document/template`
- `thesis_docs/document/THESIS_KNOWLEDGE_BASE.md`
- `thesis_docs/document/THESIS_MASTER_KNOWLEDGE.md`
- `thesis_docs/document/THESIS_DRAFT.md`
- `thesis_docs/document/Documentatie Proiect (1).docx`

#### `fast3r/` – external model code

This directory contains the model implementation that the project integrates and, for thesis purposes, is mainly useful for:

- verifying the exact Fast3R citation
- checking model configuration defaults
- understanding how training and evaluation entry points are structured

It is not the best place to start when trying to understand the project-specific contribution. For that, `src/` and `experiments/` are more important.

### Recommended reading order for a future agent

If a future agent needs to become productive quickly, the best order is:

1. `thesis_docs/document/THESIS_DRAFT.md`
2. `thesis_docs/document/THESIS_MASTER_KNOWLEDGE.md`
3. `thesis_docs/document/THESIS_INTERNAL_FILE_LEGEND.md`
4. `src/README.md`
5. `src/pipeline/fast3r_pipeline.py`
6. `src/pipeline/fast3r/frame_selection.py`
7. `src/pipeline/fast3r/scale_normalization.py`
8. `src/pipeline/fast3r/mesh_reconstruction.py`
9. `experiments/vast_training/README.md`
10. `agent/CODEX_WORKLOG.md`

### Important note

The repository inventory script is the source of truth for a complete file/folder dump.

This knowledge-base section is intentionally curated so that another agent can start navigating productively instead of reading tens of thousands of dependency or cache entries. If a full raw listing is needed, rerun:

- `python scripts/print_repo_inventory.py`
