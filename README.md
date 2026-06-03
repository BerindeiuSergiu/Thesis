# Fast3R Indoor Reconstruction Desktop Application

A practical video-to-scene pipeline for indoor reconstruction. The application takes a handheld room video, selects representative frames, reconstructs dense multiview geometry with Fast3R, normalizes the output scale, and exports an inspectable Gaussian-splat initialization with an optional mesh branch.

The main engineering objective is **fast time-to-inspectable-scene**. Traditional reconstruction and Gaussian-splat workflows commonly involve multiple iterative stages before a user can inspect a useful result. This project focuses on producing reusable scene assets from one local Fast3R reconstruction pass.

## Why This Project Exists

Indoor videos contain blur, repeated views, weakly textured walls, inconsistent exposure, and large numbers of redundant frames. Sending every frame into a reconstruction pipeline increases processing cost without guaranteeing better geometry.

This application reduces that overhead through:

- geometry-aware representative-frame selection;
- Fast3R multiview inference in one reconstruction pass;
- confidence-based point retention and view filtering;
- scale normalization for more useful real-world geometry;
- shared point-cloud cleanup before output branching;
- Gaussian-splat initialization for fast visual inspection;
- optional mesh generation for geometry-oriented workflows;
- a PyQt desktop interface with local scene history and viewing.

## Performance Positioning

The project prioritizes a short local feedback loop. Recorded Fast3R runs with Gaussian export completed in approximately:

| Selected frames | Total runtime |
| ---: | ---: |
| 80 | 148 seconds |
| 120 | 224 seconds |
| 300 | 487 seconds |

An archived internal sparse-COLMAP experiment took approximately 8 minutes for 100 sampled frames and reconstructed 24 camera poses with 1,230 sparse points. This historical result motivates the performance focus, but it is not an apples-to-apples benchmark against dense COLMAP MVS or fully optimized Gaussian-splat applications.

The current Gaussian output is an initialization for downstream Gaussian-splat workflows. It is not a fully optimized radiance-field result. A controlled cross-tool benchmark against COLMAP dense MVS, Nerfstudio Splatfacto, the INRIA 3DGS reference implementation, and Postshot remains future work.

## Application Pipeline

```text
Room video
  -> visual and geometry-aware frame selection
  -> Fast3R multiview reconstruction
  -> confidence and numeric filtering
  -> scale normalization
  -> shared point-cloud cleanup
  -> Gaussian-splat initialization
  -> optional mesh export
  -> local scene history and viewer
```

Each run is stored under `src/outputs/scene_<timestamp>/`. Runtime outputs, videos, datasets, model checkpoints, experiment results, shell launchers, and thesis-writing files are intentionally ignored by Git.

## Repository Layout

```text
src/
|- main.py                     # PyQt desktop application entry point
|- app/                        # UI, background worker, scene list
|- pipeline/
|  |- fast3r_pipeline.py       # End-to-end orchestration
|  `- fast3r/                  # Selection, inference, scaling, exports
|- viewer/                     # Local scene viewer
|- config/settings.yaml        # Runtime presets and pipeline settings
|- data/                       # Local app state, ignored except placeholder
`- outputs/                    # Generated scenes, ignored except placeholder

fast3r/                        # Vendored Fast3R implementation
experiments/
|- beta_pipeline_testing/      # Pipeline experiments and comparisons
|- geometry_test/              # Geometry-aware selection experiments
|- input_quality_lab/          # Video input diagnostics
|- gaussian_splatting_methods/ # Gaussian export experiments
|- mesh_reconstrction_methods/ # Mesh reconstruction experiments
`- vast_training/              # GA-head fine-tuning and evaluation infrastructure

data/raw/                      # Local input videos, ignored except placeholder
```

The standalone DUSt3R checkout was removed from the repository. Fast3R's internal `fast3r/fast3r/dust3r/` package remains because it is a required dependency of Fast3R.

## Run The Desktop App

Install the application dependencies:

```bash
python -m pip install -r src/requirements.txt
```

Launch from the repository root:

```bash
python -m src.main
```

The default settings are defined in `src/config/settings.yaml`. The current preset enables real Fast3R inference, geometry-aware selection, confidence filtering, scale normalization, and Gaussian output. Mesh export is available but disabled in the default high-quality preset to keep the feedback loop fast.

## Scene Outputs

A completed run can include:

- `scene_metadata.json`;
- selected frame images;
- Fast3R point-retention statistics;
- scale-normalization report;
- processed point cloud;
- Gaussian-splat `.ply` and `.npz` files;
- optional mesh `.ply` and `.obj` files;
- optional depth maps and intermediate NumPy arrays.

Generated artifacts stay local and are excluded by `.gitignore`.

## Indoor Fine-Tuning

The repository also contains a separate Vast.ai training workflow under `experiments/vast_training/`. The intended progression is:

1. load the pretrained Fast3R model;
2. fine-tune the global alignment head on indoor data;
3. evaluate the pretrained and fine-tuned checkpoints on the same validation setup;
4. export the final checkpoint for application-side experiments.

Fine-tuning infrastructure is implemented, but completed fine-tuning results are not yet presented as validated experimental findings.

## Development Notes

- Keep project-specific application changes under `src/`.
- Treat `fast3r/` as vendored model code unless an upstream adaptation is necessary.
- Keep generated outputs, datasets, videos, checkpoints, archives, thesis documents, and shell launchers out of version control.
- Use `scripts/print_repo_inventory.py` to regenerate a repository inventory when needed.

## Current Scope

The current thesis contribution is a Fast3R-based indoor reconstruction workflow with input curation, confidence-aware cleanup, practical scale normalization, shared dual-output processing, local visualization, and preparation for indoor adaptation through GA-head fine-tuning.

Future work includes controlled runtime comparisons, geometric ground-truth evaluation, completed indoor fine-tuning experiments, improved scale recovery, mesh-quality analysis, and optional Gaussian optimization for higher-fidelity view synthesis.
