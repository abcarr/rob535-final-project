# Check if it's running yet
squeue -u abcarr

# Once it starts (status changes to 'R' for Running), view output
tail -f logs/train_1epoch_test_36958883.out

# Or check the error log
tail -f logs/train_1epoch_test_36958883.err

## Post-Job Checklist

### 1. Check Job Status
```bash
squeue -u abcarr
```
- If it's not listed, the job finished (successfully or failed)

### 2. Check Job Efficiency
```bash
seff 36958883
```
- Look at CPU efficiency (ideally 50-90%)
- Look at Memory efficiency 
- Note peak memory used

### 3. Check if Job Completed Successfully
```bash
tail -50 logs/train_1epoch_test_36958883.out
```
- Should see "Epoch 0: 100%" and final validation metrics
- Look for "Saving checkpoint" message

### 4. Verify Checkpoint Was Saved
```bash
ls -lh exp/WoTE/1epoch_validation/lightning_logs/version_0/checkpoints/
```
- Should see a `.ckpt` file (likely several hundred MB)

### 5. Check for Any Errors
```bash
cat logs/train_1epoch_test_36958883.err
```
- Should be empty or just warnings (not errors)

### 6. Check Tensorboard Logs
```bash
ls -lh exp/WoTE/1epoch_validation/lightning_logs/version_0/
```
- Should see `events.out.tfevents.*` files

### 7. Review Training Metrics
```bash
grep "train_loss" logs/train_1epoch_test_36958883.out | tail -20
```
- Verify losses were decreasing

## Next Steps: Evaluation

### 1. Explore Evaluation Scripts
```bash
ls -lh scripts/evaluation/
```
- Check what evaluation scripts are available

### 2. Read Evaluation Script
```bash
cat scripts/evaluation/eval_wote.sh
```
- Understand what it does and what paths need updating

### 3. Check if Metric Cache is Needed
```bash
# May need to run metric caching first
ls -lh /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/cache/
```

### 4. Create Evaluation SLURM Script
```bash
# Create slurm_jobs/eval_1epoch_test.slurm
# Should be much faster than training (15-30 min estimate)
# Single GPU, fewer CPUs needed
```

### 5. Test Evaluation Run
```bash
sbatch slurm_jobs/eval_1epoch_test.slurm
```

### 6. After Evaluation Success, Launch Full Training
```bash
# Create slurm_jobs/train_20epochs.slurm
# Time: 32:00:00 (30 hours + buffer)
# Cost: ~$5.26
sbatch slurm_jobs/train_20epochs.slurm
```

## Checkpoint Fix

The first training run didn't save checkpoints. Running new test with checkpointing enabled:

```bash
sbatch slurm_jobs/train_1epoch_with_checkpoint.slurm
```

This should create checkpoint files in:
`exp/WoTE/1epoch_with_checkpoint/lightning_logs/version_*/checkpoints/`

---

## ConvGRU Temporal Fusion (Branch: abcarr-convgru)

### Goal
Add temporal BEV fusion using ConvGRU to enable multi-frame history aggregation with ego-motion compensation.

### Implementation Progress

#### Phase 0: Setup ✅
- [x] Created branch `abcarr-convgru` based on `abcarr`
- [x] Created `navsim/agents/WoTE/modules/temporal_fusion.py` with:
  - ConvGRUCell implementation
  - ConvGRU multi-layer module
  - TemporalBEVFusion wrapper with ego-motion warping

#### Next Steps
- [ ] Add `use_convgru` parameter to `configs/default.py`
- [ ] Update WoTE model to conditionally use temporal fusion
- [ ] Update feature builder for multi-frame loading
- [ ] Test with dummy data
- [ ] Run 1-epoch test on HPC

See `CONVGRU_INTEGRATION_PLAN.md` for full implementation plan.

