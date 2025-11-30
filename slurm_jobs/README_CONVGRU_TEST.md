# ConvGRU 1-Epoch Test Job

## Purpose
Test the ConvGRU temporal BEV fusion implementation on HPC with real WoTE data for 1 epoch.

## Configuration

### ConvGRU Settings
- `use_convgru=true` - Enable temporal fusion
- `num_history_frames=4` - Use 4 historical frames
- `slice_indices=[0,1,2,3]` - Chronological frame sequence
- `temporal_hidden_dim=512` - ConvGRU hidden dimension

### Resources
- **Memory**: 40GB (increased from 32GB for 4-frame processing)
- **Time**: 3 hours (vs 2 hours for baseline, due to 4× backbone passes)
- **Batch Size**: 8 (reduced from 16 to manage memory)
- **GPUs**: 1 (account limit)

## What This Test Verifies

### ✅ Critical Checks
1. **Multi-frame data loading** works with real NAVSIM data
2. **Ego-motion computation** produces valid transforms
3. **BEV warping** aligns historical frames correctly
4. **ConvGRU module** processes temporal sequences
5. **Memory usage** stays under 40GB
6. **Training completes** without crashes
7. **Checkpoints save** correctly

### 📊 Expected Behavior
- Training should complete in ~2.5-3 hours
- GPU memory usage: ~35-38GB (vs 28GB baseline)
- Training speed: ~0.6-0.7 it/s (vs ~0.95 it/s baseline)
- Loss should decrease (sanity check)
- No NaN/Inf in losses

## How to Run

### 1. On HPC, ensure you're on the correct branch:
```bash
cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project
git checkout abcarr-convgru
git pull origin abcarr-convgru
```

### 2. Submit the job:
```bash
sbatch slurm_jobs/train_1epoch_convgru_test.slurm
```

### 3. Monitor the job:
```bash
# Check job status
squeue -u abcarr

# Watch output
tail -f logs/train_convgru_1epoch_<JOB_ID>.out

# Check for errors
tail -f logs/train_convgru_1epoch_<JOB_ID>.err
```

### 4. After completion, check results:
```bash
# Get job statistics
seff <JOB_ID>

# Check if checkpoint was saved
ls -lh exp/WoTE/convgru_1epoch_test/lightning_logs/version_*/checkpoints/

# Review final output
tail -50 logs/train_convgru_1epoch_<JOB_ID>.out
```

## Troubleshooting

### If job fails with OOM (Out of Memory):
```bash
# Edit the SLURM file and reduce batch size:
dataloader.params.batch_size=4  # Was 8
```

### If training is too slow:
This is expected! ConvGRU processes 4× more frames through the backbone.
- Baseline: 1 frame per sample
- ConvGRU: 4 frames per sample
- Expected slowdown: ~2-3× (due to parallel processing within batch)

### If loss is NaN or Inf:
Check logs for:
```bash
grep -i "nan\|inf" logs/train_convgru_1epoch_<JOB_ID>.out
```
This could indicate:
- Gradient explosion (check learning rate)
- BEV warping error (check ego-motion transforms)
- ConvGRU instability (may need gradient clipping)

## Success Criteria

### ✅ Job passes if:
1. Job completes without crash
2. Checkpoint file exists
3. GPU memory < 40GB
4. Loss decreases over training
5. No NaN/Inf in losses
6. No shape mismatch errors

### Next Steps After Success:
1. Compare 1-epoch checkpoint performance vs baseline
2. Run 5-epoch validation test
3. Launch full 20-epoch training
4. Run ablation studies

## Cost Estimate
- 1 epoch: ~3 hours × $0.33/2hr ≈ $0.50
- 20 epochs: ~60 hours × $0.33/2hr ≈ $10

## Related Files
- Implementation: `navsim/agents/WoTE/WoTE_features.py`
- Model integration: `navsim/agents/WoTE/WoTE_model.py`
- Config: `navsim/agents/WoTE/configs/default.py`
- Tests: `test_feature_builder.py`, `test_temporal_fusion_only.py`
- Docs: `CONVGRU_INTEGRATION_PLAN.md`, `VERIFICATION_RESULTS.md`
