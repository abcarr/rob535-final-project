# ConvGRU Temporal BEV Fusion Integration Plan

## Overview
Integrate ConvGRU-based temporal BEV fusion into WoTE to enable multi-frame history aggregation through spatial alignment and recurrent updates. This will improve motion understanding while remaining fully differentiable.

## Current State Analysis

### WoTE Architecture
- **Current BEV generation**: `TransfuserBackbone` processes single-frame camera + LiDAR → BEV features (512 channels, 8×8 spatial)
- **Downscaling**: BEV features downscaled to 256 channels via `_bev_downscale`
- **slice_indices=[3]**: Currently uses only frame index 3 (single timestep)
- **No temporal aggregation**: Each frame processed independently

### Key Files
1. `navsim/agents/WoTE/WoTE_model.py` - Main model architecture
2. `navsim/agents/WoTE/WoTE_agent.py` - Agent wrapper
3. `navsim/agents/WoTE/WoTE_features.py` - Feature extraction
4. `navsim/agents/transfuser/transfuser_backbone.py` - BEV generation backbone

## Proposed Architecture

### Temporal BEV Fusion Module
```
Multi-frame Input (t-n, ..., t-1, t)
         ↓
   TransfuserBackbone (per-frame)
         ↓
   BEV Features [B, T, 512, 8, 8]
         ↓
   Ego-motion Compensation (Affine Warp)
         ↓
   BEV Features (aligned) [B, T, 512, 8, 8]
         ↓
   ConvGRU (temporal fusion)
         ↓
   Fused BEV [B, 512, 8, 8]
         ↓
   Rest of WoTE pipeline
```

## Implementation Plan

### Phase 1: Setup & Infrastructure ✅
**Status**: Already done, waiting for checkpoint validation

### Phase 2: Multi-Frame Data Loading (Week 1)
**Goal**: Enable loading of temporal sequences instead of single frames

#### 2.1 Update slice_indices Configuration
**File**: `navsim/agents/WoTE/configs/default.py`
```python
# Change from:
slice_indices=[3]

# To (for 4-frame history):
slice_indices=[0, 1, 2, 3]  # indices 0-3 for temporal sequence
```

#### 2.2 Modify Feature Builder
**File**: `navsim/agents/WoTE/WoTE_features.py`

**Current**: Processes single frame at `slice_indices[0]`
**New**: Process all frames in `slice_indices`

Changes needed:
- Remove assertion `assert len(self.slice_indices) == 1`
- Loop over all slice indices to extract multi-frame features
- Stack temporal features: `[B, T, C, H, W]`
- Extract ego-motion between frames for warping

**Key Method**: `compute_features(scene: Scene)`
```python
# Pseudocode
def compute_features(self, scene: Scene):
    camera_frames = []
    lidar_frames = []
    ego_states = []
    
    for idx in self.slice_indices:
        camera_frames.append(scene.frames[idx].camera)
        lidar_frames.append(scene.frames[idx].lidar)
        ego_states.append(scene.frames[idx].ego_state)
    
    # Compute relative ego-motion transformations
    ego_motions = self._compute_ego_motion_transforms(ego_states)
    
    return {
        "camera_feature": torch.stack(camera_frames),  # [T, C, H, W]
        "lidar_feature": torch.stack(lidar_frames),    # [T, C, H, W]
        "ego_motion": ego_motions,                      # [T-1, 3, 3] affine matrices
        "status_feature": ego_states[-1],               # Use current frame
    }
```

#### 2.3 Ego-Motion Transform Computation
**New Method**: `_compute_ego_motion_transforms()`

Compute affine transformation matrices to warp previous BEV frames to current frame:
```python
def _compute_ego_motion_transforms(self, ego_states):
    """
    Compute 2D affine transformation matrices for BEV warping.
    
    Args:
        ego_states: List of ego poses [(x, y, heading), ...]
    
    Returns:
        Affine matrices [T-1, 2, 3] to warp frames [0..T-2] → frame [T-1]
    """
    current_pose = ego_states[-1]  # Reference frame
    transforms = []
    
    for prev_pose in ego_states[:-1]:
        # Compute relative transformation
        delta_x = current_pose.x - prev_pose.x
        delta_y = current_pose.y - prev_pose.y
        delta_theta = current_pose.heading - prev_pose.heading
        
        # Build 2D affine matrix for BEV warping
        # Accounts for rotation + translation
        cos_theta = np.cos(delta_theta)
        sin_theta = np.sin(delta_theta)
        
        affine_matrix = np.array([
            [cos_theta, -sin_theta, delta_x],
            [sin_theta,  cos_theta, delta_y]
        ])
        transforms.append(torch.tensor(affine_matrix))
    
    return torch.stack(transforms)
```

### Phase 3: ConvGRU Integration (Week 2)

#### 3.1 Copy ConvGRU Code
**New File**: `navsim/agents/WoTE/modules/temporal_fusion.py`

Copy ConvGRU implementation and add wrapper:
```python
from ConvGRU_pytorch.convGRU import ConvGRU, ConvGRUCell

class TemporalBEVFusion(nn.Module):
    def __init__(self, 
                 bev_channels=512, 
                 hidden_dim=512,
                 bev_size=(8, 8),
                 num_history=4):
        super().__init__()
        
        self.bev_size = bev_size
        self.num_history = num_history
        
        # ConvGRU for temporal fusion
        self.conv_gru = ConvGRU(
            input_size=bev_size,
            input_dim=bev_channels,
            hidden_dim=[hidden_dim],
            kernel_size=(3, 3),
            num_layers=1,
            dtype=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
    def warp_bev_features(self, bev_features, ego_motion):
        """
        Warp previous BEV frames to current frame using ego-motion.
        
        Args:
            bev_features: [B, T, C, H, W]
            ego_motion: [B, T-1, 2, 3] affine matrices
        
        Returns:
            warped_features: [B, T, C, H, W] (all aligned to frame T)
        """
        B, T, C, H, W = bev_features.shape
        warped = []
        
        # Current frame doesn't need warping
        warped.append(bev_features[:, -1])
        
        # Warp historical frames
        for t in range(T-1):
            prev_bev = bev_features[:, t]  # [B, C, H, W]
            affine = ego_motion[:, t]       # [B, 2, 3]
            
            # Create sampling grid
            grid = F.affine_grid(affine, prev_bev.size(), align_corners=False)
            
            # Warp using bilinear interpolation
            warped_bev = F.grid_sample(prev_bev, grid, 
                                       mode='bilinear', 
                                       padding_mode='zeros',
                                       align_corners=False)
            warped.insert(0, warped_bev)
        
        return torch.stack(warped, dim=1)  # [B, T, C, H, W]
    
    def forward(self, bev_features, ego_motion, hidden_state=None):
        """
        Args:
            bev_features: [B, T, C, H, W] - Multi-frame BEV features
            ego_motion: [B, T-1, 2, 3] - Ego-motion transformations
            hidden_state: Optional previous hidden state for stateful operation
        
        Returns:
            fused_bev: [B, C, H, W] - Temporally fused BEV feature
            hidden_state: Hidden state for next iteration
        """
        # 1. Warp historical BEVs to current frame
        aligned_bevs = self.warp_bev_features(bev_features, ego_motion)
        
        # 2. Apply ConvGRU temporal fusion
        layer_output_list, hidden_states = self.conv_gru(aligned_bevs, hidden_state)
        
        # 3. Return fused BEV (last timestep of output sequence)
        fused_bev = layer_output_list[0][:, -1]  # [B, C, H, W]
        
        return fused_bev, hidden_states
```

#### 3.2 Integrate into WoTE Model
**File**: `navsim/agents/WoTE/WoTE_model.py`

**In `__init__`**:
```python
# Add after transfuser backbone initialization
self.config = config
self.use_convgru = config.use_convgru if hasattr(config, 'use_convgru') else False

# Conditionally create temporal fusion module
if self.use_convgru:
    from navsim.agents.WoTE.modules.temporal_fusion import TemporalBEVFusion
    
    self.temporal_bev_fusion = TemporalBEVFusion(
        bev_channels=512,
        hidden_dim=config.temporal_hidden_dim,
        bev_size=(8, 8),
        num_history=config.num_history_frames
    )
    print(f"✓ ConvGRU temporal fusion enabled with {config.num_history_frames} frames")
else:
    self.temporal_bev_fusion = None
    print("✓ Using single-frame BEV (ConvGRU disabled)")
```

**In `_process_backbone_features`**:
```python
def _process_backbone_features(self, camera_feature, lidar_feature, ego_motion=None):
    """
    Process backbone with optional temporal fusion.
    
    Args:
        camera_feature: [B, T, C, H, W] if use_convgru=True, else [B, C, H, W]
        lidar_feature: [B, T, C, H, W] if use_convgru=True, else [B, C, H, W]
        ego_motion: [B, T-1, 2, 3] affine matrices (only if use_convgru=True)
    """
    is_temporal = camera_feature.ndim == 5
    
    # Branch 1: Temporal fusion with ConvGRU (if enabled)
    if self.use_convgru and is_temporal:
        B, T, C, H, W = camera_feature.shape
        
        # Process each frame through backbone
        bev_features_list = []
        for t in range(T):
            _, bev_t, _ = self._backbone(
                camera_feature[:, t],
                lidar_feature[:, t]
            )
            bev_features_list.append(bev_t)
        
        # Stack temporal BEVs: [B, T, 512, 8, 8]
        bev_features_temporal = torch.stack(bev_features_list, dim=1)
        
        # Apply temporal fusion with ego-motion compensation
        backbone_bev_feature, _ = self.temporal_bev_fusion(
            bev_features_temporal, 
            ego_motion
        )
    
    # Branch 2: Single-frame processing (original behavior)
    else:
        # Handle both [B, C, H, W] and [B, T, C, H, W] where T=1
        if is_temporal:
            camera_feature = camera_feature[:, -1]  # Use last frame
            lidar_feature = lidar_feature[:, -1]
        
        _, backbone_bev_feature, _ = self._backbone(camera_feature, lidar_feature)
    
    # Continue with existing downscaling (same for both branches)
    bev_feature = self._bev_downscale(backbone_bev_feature).flatten(-2, -1).permute(0, 2, 1)
    flatten_bev_feature = bev_feature + self._keyval_embedding.weight[None, :, :]
    
    return backbone_bev_feature, flatten_bev_feature
```

### Phase 4: Configuration & Training (Week 3)

#### 4.1 Update Default Config (Backward Compatible)
**File**: `navsim/agents/WoTE/configs/default.py`

Add temporal fusion parameters with default values (disabled by default):

```python
@dataclass
class WoTEConfig:
    # ... existing config params ...
    
    # Temporal BEV Fusion (ConvGRU) - CONFIGURABLE
    use_convgru: bool = False  # ← Main toggle for temporal fusion
    num_history_frames: int = 4
    temporal_hidden_dim: int = 512
    temporal_kernel_size: tuple = (3, 3)
    temporal_num_layers: int = 1
    temporal_fusion_lr_mult: float = 1.0  # Learning rate multiplier for ConvGRU
```

**Benefits of this approach**:
- ✅ Single config file (no separate temporal_fusion.py)
- ✅ Easy A/B testing: just set `agent.config.use_convgru=true`
- ✅ Backward compatible: defaults to False (original behavior)
- ✅ Can switch at training time via Hydra CLI

#### 4.2 Create Temporal Config (Optional Alternative)
**New File**: `navsim/agents/WoTE/configs/temporal_fusion.py` *(Optional)*

For users who want a pre-configured temporal setup:

```python
from navsim.agents.WoTE.configs.default import WoTEConfig
from dataclasses import dataclass

@dataclass
class TemporalWoTEConfig(WoTEConfig):
    """WoTE with temporal BEV fusion using ConvGRU - Pre-configured."""
    
    # Enable temporal fusion
    use_convgru: bool = True  # ← Enabled by default in this config
    num_history_frames: int = 4
    slice_indices: list = (0, 1, 2, 3)  # 4-frame history
    
    # ConvGRU settings
    temporal_hidden_dim: int = 512
    temporal_kernel_size: tuple = (3, 3)
    temporal_num_layers: int = 1
```

#### 4.3 Training Script Examples

**Option A: Enable via CLI (Recommended)**
```bash
#!/bin/bash
# scripts/training/run_wote_temporal.sh

export PYTHONPATH=/path/to/repo
export NUPLAN_MAPS_ROOT="..."
# ... other exports

# Enable ConvGRU via command line flag
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_convgru=true \
agent.config.num_history_frames=4 \
experiment_name=WoTE/temporal_fusion \
scene_filter=navtrain \
dataloader.params.batch_size=8 \
trainer.params.max_epochs=20 \
split=trainval
```

**Option B: Use Pre-configured Temporal Config**
```bash
# Alternative: Use TemporalWoTEConfig (if you created it)
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.temporal_fusion.TemporalWoTEConfig \
experiment_name=WoTE/temporal_fusion \
scene_filter=navtrain \
dataloader.params.batch_size=8 \
trainer.params.max_epochs=20 \
split=trainval
```

**Option C: Baseline (Disable ConvGRU)**
```bash
# Original behavior - single frame, no temporal fusion
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_convgru=false \
experiment_name=WoTE/baseline \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=20 \
split=trainval
```

**Note**: Reduce batch size (16→8) when `use_convgru=true` due to increased memory

#### 4.4 SLURM Scripts with use_convgru Toggle

**File**: `slurm_jobs/train_temporal_wote.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=wote_temporal
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB                    # Increased from 32GB
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00               # Slightly longer due to added computation
#SBATCH --output=logs/train_temporal_%j.out
#SBATCH --error=logs/train_temporal_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

# ... environment setup same as before ...

# Enable ConvGRU temporal fusion
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_convgru=true \
agent.config.num_history_frames=4 \
experiment_name=WoTE/temporal_fusion \
scene_filter=navtrain \
dataloader.params.batch_size=8 \
trainer.params.max_epochs=20 \
trainer.params.check_val_every_n_epoch=1 \
split=trainval
```

**File**: `slurm_jobs/train_baseline_wote.slurm`
```bash
#!/bin/bash
# Same SLURM directives but with use_convgru=false

python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_convgru=false \
experiment_name=WoTE/baseline \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=20 \
trainer.params.check_val_every_n_epoch=1 \
split=trainval
```

**Easy A/B Testing**:
```bash
# Test temporal fusion
sbatch slurm_jobs/train_temporal_wote.slurm

# Test baseline
sbatch slurm_jobs/train_baseline_wote.slurm

# Compare results later
```

### Phase 5: Testing & Validation (Week 4)

#### 5.1 Unit Tests
**New File**: `tests/test_temporal_fusion.py`

```python
import torch
from navsim.agents.WoTE.modules.temporal_fusion import TemporalBEVFusion

def test_temporal_fusion():
    B, T, C, H, W = 2, 4, 512, 8, 8
    
    # Create module
    fusion = TemporalBEVFusion(
        bev_channels=512,
        hidden_dim=512,
        bev_size=(8, 8),
        num_history=4
    )
    
    # Fake input
    bev_features = torch.randn(B, T, C, H, W)
    ego_motion = torch.randn(B, T-1, 2, 3)
    
    # Forward pass
    fused_bev, hidden = fusion(bev_features, ego_motion)
    
    # Check output shape
    assert fused_bev.shape == (B, C, H, W)
    print("✓ Temporal fusion test passed")

def test_ego_motion_warping():
    # Test that warping preserves features when ego-motion is identity
    fusion = TemporalBEVFusion()
    
    B, T, C, H, W = 1, 4, 512, 8, 8
    bev = torch.randn(B, T, C, H, W)
    
    # Identity transformation
    identity = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]).repeat(B, T-1, 1, 1)
    
    warped = fusion.warp_bev_features(bev, identity)
    
    # Should be approximately equal
    assert torch.allclose(warped, bev, atol=1e-5)
    print("✓ Ego-motion warping test passed")

if __name__ == "__main__":
    test_temporal_fusion()
    test_ego_motion_warping()
```

#### 5.2 Small-Scale Training Test
Before full training, run 1-epoch test:

```bash
sbatch slurm_jobs/train_temporal_1epoch_test.slurm
```

Verify:
- Data loads correctly (4 frames per sample)
- GPU memory usage (~35GB)
- Training converges
- Losses decrease

#### 5.3 Visualization
Add visualization to verify temporal alignment:

```python
def visualize_temporal_alignment(bev_features, warped_features, save_path):
    """Visualize BEV features before/after warping."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for t in range(4):
        # Original
        axes[0, t].imshow(bev_features[0, t, 0].cpu().numpy())
        axes[0, t].set_title(f'Original t={t}')
        
        # Warped
        axes[1, t].imshow(warped_features[0, t, 0].cpu().numpy())
        axes[1, t].set_title(f'Warped t={t}')
    
    plt.savefig(save_path)
```

### Phase 6: Ablation Studies (Week 5)

Compare different configurations:

#### 6.1 Baseline vs Temporal
All controlled by `use_convgru` flag:

- **Baseline**: `use_convgru=false` (single frame)
- **Temporal-2**: `use_convgru=true num_history_frames=2`
- **Temporal-4**: `use_convgru=true num_history_frames=4` (proposed)
- **Temporal-8**: `use_convgru=true num_history_frames=8` (if memory allows)

Easy to launch multiple experiments:
```bash
# Baseline
sbatch slurm_jobs/train_baseline.slurm

# Temporal variants
for n_frames in 2 4 8; do
  sbatch slurm_jobs/train_temporal_${n_frames}frames.slurm
done
```

#### 6.2 Architecture Variants
- **ConvGRU-1layer**: 1 layer (proposed)
- **ConvGRU-2layer**: 2 layers (deeper)
- **ConvGRU-no-warp**: No ego-motion compensation (ablation)

#### 6.3 Evaluation Metrics
Track improvements in:
- PDM Score (overall)
- no_at_fault_collisions (motion understanding)
- time_to_collision_within_bound (anticipation)
- ego_progress (planning quality)

Expected improvement: 5-10% PDM score increase

## Implementation Checklist

### Prerequisites ✅
- [x] Environment setup complete
- [x] Training infrastructure working
- [x] Checkpoint saving verified

### Configuration Setup (Before Week 1)
- [ ] Add `use_convgru` parameter to `configs/default.py`
- [ ] Add related temporal params (`num_history_frames`, etc.)
- [ ] Set defaults: `use_convgru=False` for backward compatibility
- [ ] Test that baseline still works with new params

### Week 1: Data Loading
- [ ] Update `slice_indices` config to `[0, 1, 2, 3]`
- [ ] Modify `WoTE_features.py` to handle multi-frame extraction
- [ ] Implement `_compute_ego_motion_transforms()`
- [ ] Test data loading with temporal sequences
- [ ] Verify ego-motion computations are correct

### Week 2: ConvGRU Integration
- [ ] Create `modules/temporal_fusion.py` with ConvGRU
- [ ] Implement `TemporalBEVFusion` module
- [ ] Implement BEV warping with `F.affine_grid`
- [ ] Integrate into `WoTE_model._process_backbone_features()`
- [ ] Test forward pass with dummy data
- [ ] Verify gradient flow through ConvGRU

### Week 3: Configuration & Training
- [ ] Create `configs/temporal_fusion.py`
- [ ] Create training scripts for temporal model
- [ ] Create SLURM job for 1-epoch test
- [ ] Run 1-epoch test and verify:
  - [ ] Model initializes correctly
  - [ ] Training runs without errors
  - [ ] GPU memory usage acceptable
  - [ ] Losses decrease

### Week 4: Full Training
- [ ] Launch 20-epoch training job
- [ ] Monitor training progress
- [ ] Check for overfitting/underfitting
- [ ] Save best checkpoints

### Week 5: Evaluation & Analysis
- [ ] Run evaluation on test set
- [ ] Compare PDM scores: baseline vs temporal
- [ ] Ablation studies (2/4/8 frames, with/without warping)
- [ ] Visualize temporal alignment
- [ ] Generate performance report

## Memory & Compute Estimates

### Memory Requirements
- **Baseline WoTE**: ~28GB (from your first run)
- **Temporal WoTE (4 frames)**: ~35-40GB estimated
  - 4× backbone forward passes per sample
  - ConvGRU hidden state storage
  - Warped BEV features storage

**Recommendation**: Request 40GB mem in SLURM script

### Training Time
- **Baseline**: ~1.5 hrs/epoch
- **Temporal (estimated)**: ~2.5-3 hrs/epoch
  - 4× more backbone computations
  - + ConvGRU overhead
  - + Warping overhead

**20-epoch estimate**: 50-60 hours (~$15-20 cost)

### Batch Size Adjustment
- **Baseline**: batch_size=16
- **Temporal**: batch_size=8 (recommended)
  - Reduces memory pressure
  - Still maintains reasonable training speed

## Potential Challenges & Solutions

### Challenge 1: Memory Overflow
**Symptom**: CUDA out of memory errors
**Solutions**:
- Reduce batch size to 4
- Use gradient checkpointing for backbone
- Process temporal frames sequentially instead of in parallel

### Challenge 2: Ego-Motion Estimation Errors
**Symptom**: Warped BEVs look misaligned
**Solutions**:
- Verify ego-motion transformation math
- Visualize warped features (add debug visualizations)
- Check coordinate frame conventions (NuPlan uses specific convention)

### Challenge 3: Training Instability
**Symptom**: Loss spikes or divergence
**Solutions**:
- Lower learning rate for ConvGRU module
- Add gradient clipping
- Initialize ConvGRU weights carefully
- Use layer normalization in ConvGRU

### Challenge 4: No Performance Gain
**Symptom**: Temporal model performs same as baseline
**Solutions**:
- Verify temporal fusion is actually being used (add logging)
- Check if warping is working correctly
- Try longer history (8 frames instead of 4)
- Increase ConvGRU capacity (2 layers, more hidden dim)

## Expected Outcomes

### Performance Improvements
- **PDM Score**: 5-10% increase expected
- **Motion Understanding**: Better handling of dynamic agents
- **Trajectory Planning**: Smoother, more anticipatory trajectories

### Qualitative Improvements
- Better handling of:
  - Moving vehicles (tracking their motion)
  - Intersection scenarios (anticipating cross-traffic)
  - Lane changes (understanding ego and neighbor intent)
  - Occlusions (temporal context helps fill gaps)

### Failure Cases
Even with temporal fusion, expect challenges with:
- Rare/novel scenarios (still limited by training data)
- Very long-term predictions (>8 seconds)
- Sudden, unpredictable agent behaviors

## Next Steps After Implementation

1. **Paper/Report**: Document improvements and methodology
2. **Hyperparameter Tuning**: Optimize num_history, hidden_dim, etc.
3. **Ensemble**: Combine baseline + temporal models
4. **Attention Mechanism**: Add temporal attention instead of just ConvGRU
5. **End-to-End Finetuning**: Finetune entire pipeline with temporal fusion

## References

- ConvGRU Paper: "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" (Shi et al., 2015)
- Temporal Fusion in Autonomous Driving: See BEVFormer, FIERY, etc.
- Ego-Motion Compensation: Standard practice in multi-view 3D perception

## Questions to Consider

1. **How many history frames?** Start with 4, experiment with 2/8
2. **Freeze backbone initially?** Could help ConvGRU learn faster
3. **Pretrain temporal module?** Consider pretraining on reconstruction task
4. **Use optical flow?** Alternative to ego-motion-based warping
5. **Asymmetric temporal fusion?** Weight recent frames more heavily

---

**Status**: Ready to begin implementation after checkpoint validation completes
**Estimated Total Time**: 5 weeks
**Estimated Cost**: $20-25 (including ablations)
**Expected Performance Gain**: 5-10% PDM score improvement
