# Pre-Rent Checklist

Do not rent the GPU until these are true.

## Code readiness

- [ ] `convert_hf_fast3r_to_lightning_ckpt.py` has run successfully
- [ ] the converted `.ckpt` file exists and is readable
- [ ] the quick debug command from `launch_training.py --config-name quick_debug` has been printed and reviewed
- [ ] the smoke-test training loop has completed at least once
- [ ] the head-freeze callback logs only the parameter groups you intended to train
- [ ] the ScanNet root on the remote machine matches the expected loader layout

## Training scope

- [ ] you have chosen the first paid config: `scannet_3day_finetune` or `final_run`
- [ ] you have chosen the first training stage: `ga_head`, `ga_head_plus_local`, or `decoder_plus_ga_head`
- [ ] you have chosen the exact ScanNet split/layout you will train on
- [ ] you know the exact dataset root path on the rented machine
- [ ] you know where logs and checkpoints will be written

## Cost controls

- [ ] first run will use an `on-demand` instance, not interruptible
- [ ] checkpointing is set to every epoch or better
- [ ] batch size is conservative
- [ ] max epochs are capped
- [ ] you have a hard stop budget in mind before launching

## Data safety

- [ ] important outputs will be copied off-machine after the run
- [ ] you understand that stopped instances can still incur storage charges
- [ ] you will destroy the instance when done if you do not need the storage anymore

## Practical first-run recommendation

If you want the lowest-risk first paid experiment:

- config: `scannet_3day_finetune`
- stage: `ga_head`
- rental type: `on-demand`
- batch size: `1`
- precision: `16-mixed`
- checkpoint every: `1 epoch`
