# Vast.ai LoRA Run Commands

Use this runbook for the second training stage: load the completed GA-head
checkpoint, add LoRA adapters to the Fast3R decoder attention layers, fine-tune
on the restricted ARKitScenes subset, and export merged Fast3R weights.

Replace the Vast IP address, SSH port, and GA-head checkpoint path if your
machine uses different values.

## 1. Local machine: upload the code

Run this once from Git Bash on the local machine:

```bash
cd /d/GitRepos/Thesis

export VAST_HOST=root@YOUR_VAST_IP
export VAST_SSH_PORT=YOUR_VAST_SSH_PORT

bash experiments/vast_training/vast_upload_repo.ksh "$VAST_HOST" /workspace
```

After the upload completes, connect to the remote machine using the SSH command
shown by Vast.ai.

## 2. Remote machine: select the Python environment

Run the remaining commands on the Vast.ai machine:

```bash
export PYTHON_BIN=/workspace/venvs/fast3r-py312/bin/python
export PYTHONPATH=/workspace${PYTHONPATH:+:$PYTHONPATH}

test -x "$PYTHON_BIN" || {
  echo "Missing Python environment: $PYTHON_BIN"
  exit 1
}
```

## 3. Remote machine: verify the required existing files

The LoRA run starts from the completed GA-head checkpoint. Update
`LORA_BASE_CKPT` below if your checkpoint has a different path.

```bash
export LORA_BASE_CKPT=/workspace/logs/vast_finetune_arkitscenes/runs/arkitscenes_10hour_ga_head/checkpoints/last.ckpt

test -e "$LORA_BASE_CKPT" || {
  echo "Missing GA-head checkpoint: $LORA_BASE_CKPT"
  exit 1
}

test -f /workspace/datasets/arkitscenes_processed/Training/all_metadata.npz || {
  echo "Missing processed ARKitScenes Training metadata"
  exit 1
}

test -f /workspace/datasets/arkitscenes_processed/Test/all_metadata.npz || {
  echo "Missing processed ARKitScenes Test metadata"
  exit 1
}
```

## 4. Remote machine: configure the LoRA run

```bash
export CONFIG_NAME=arkitscenes_lora_decoder_small
export RUN_NAME=arkitscenes_lora_decoder_small
export TRAIN_STAGE=lora_decoder_attention
export LORA_ENABLED=1

export DATASET_NAME=arkitscenes
export DATASET_ROOT=/workspace/datasets/arkitscenes_processed

export LR=0.0001
export MAX_STEPS=1000

export DO_SETUP=1
export DO_DOWNLOAD_DATASET=0
export DO_PREPARE_DATASET=0
export DO_TRAIN=1
export DO_EVALUATE=1
export DO_EXPORT_CHECKPOINT=1
export ENABLE_OOM_FALLBACK=1
```

The profile uses the restricted training subset configured in
`configs/arkitscenes_lora_decoder_small.json`:

- `4,000` training samples
- `50` validation samples
- LoRA rank `8`
- LoRA alpha `16`
- LoRA dropout `0.05`
- decoder attention targets: `decoder.dec_blocks.*.attn.(qkv|proj)`

## 5. Remote machine: run LoRA fine-tuning

```bash
bash /workspace/vast_training/vast_run_training.ksh /workspace
```

The script installs the LoRA dependency, materializes the GA-head checkpoint if
needed, trains the adapters, evaluates the resulting checkpoint, merges the
adapters into ordinary Fast3R weights, and bundles the outputs.

## 6. Remote machine: verify the exported result

```bash
ls -lah /workspace/TO_BE_DOWNLOADED_TO_LOCAL_MACHINE/arkitscenes_lora_decoder_small
ls -lah /workspace/checkpoints/arkitscenes_lora_decoder_small_hf_export
```

The thesis-ready result bundle is:

```text
/workspace/TO_BE_DOWNLOADED_TO_LOCAL_MACHINE/arkitscenes_lora_decoder_small/
```

The merged Hugging Face-style model directory is:

```text
/workspace/checkpoints/arkitscenes_lora_decoder_small_hf_export/
```

The merged export can be loaded by the desktop application through the normal
`Fast3R.from_pretrained(...)` path.
