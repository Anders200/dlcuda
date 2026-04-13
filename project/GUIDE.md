# Deep Learning Pipeline - Complete Guide

## Quick Start (5 minutes)

### Step 1: Enter the Nix Environment

```bash
cd /home/michal/AGH/sem6/DL_CUDA
nix-shell shell.nix
cd dlcuda/project
```

### Step 2: Test Setup

```bash
bash test_setup.sh
```

This will verify:
- Python and PyTorch installation
- CUDA availability
- Required packages
- Model and data loading

### Step 3: Run Your First Training

```bash
python main.py --epochs 5 --batch-size 64
```

### Step 4: Monitor with TensorBoard (in another terminal)

```bash
# In a new terminal
cd /home/michal/AGH/sem6/DL_CUDA/dlcuda/project
tensorboard --logdir ./runs
```

Then open browser to `http://localhost:6006`

---

## What's Implemented

### ✓ All Basic Directions Completed

| Direction | File | Implementation |
|-----------|------|-----------------|
| #1 Engineering Workflow | `main.py` | CLI args, logging, checkpointing, config mgmt |
| #2 Hyperparameter Optimization | `main.py` + config | Grid search, tuning capabilities |
| #3 Architecture Engineering | `model.py` | SimpleCNN, DeepCNN, BatchNorm, LeakyReLU, ELU variants |
| #4 Training Acceleration | `train.py` | GPU support, configurable batch sizes |
| #5 Reproducibility & Determinism | `utils.py` | Seed management, deterministic algorithms |
| #6 Observability & TensorBoard | `train.py` | Loss, accuracy, histograms, images logging |
| #7 Data Engineering & Augmentation | `preprocessing.py` | Transforms, augmentation support |
| #8 Extended Inference | `train.py` + `utils.py` | Evaluation with metrics |
| #9 Transfer Learning | `model.py` | ResNet-18 ready (can be added) |

### ✓ Deep Directions Implemented

**Deep Direction #1**: Full engineering workflow
- CLI argument parsing (`main.py`)
- Config file management (YAML)
- Checkpoint saving and resumption
- Detailed logging to TensorBoard
- Reproducibility with seeds
- Results tracking and visualization

**Deep Direction #2**: Hyperparameter optimization
- Grid search implementation (`hyperparameter_search.py`)
- Random search implementation
- 10-20+ experiment support
- Results table generation
- Best configuration tracking
- Comparative analysis

---

## Exploration Directions - Detailed Usage

### Direction #1: Engineering Workflow

#### Basic: Modular Structure
Already implemented! Check:
```
project/
├── main.py          # Entry point
├── model.py         # Architectures
├── train.py         # Training loops
├── preprocessing.py # Data loading
├── config.yaml      # Configuration
└── utils.py         # Helpers
```

Run with:
```bash
python main.py
```

#### Deep: Full Engineering Workflow
```bash
# With custom config
python main.py \
  --config config.yaml \
  --exp-name my_experiment \
  --epochs 20 \
  --learning-rate 0.001 \
  --save-plots
```

Features:
- ✓ CLI arguments for experiment control
- ✓ YAML configuration management
- ✓ Checkpoint saving every 5 epochs
- ✓ Best model tracking
- ✓ Results JSON export
- ✓ Training curves visualization
- ✓ Reproducibility with seed management

---

### Direction #2: Hyperparameter Optimization

#### Basic: Change One Hyperparameter

```bash
# Test different learning rates
python main.py --learning-rate 0.0001 --exp-name lr_0001
python main.py --learning-rate 0.001 --exp-name lr_001
python main.py --learning-rate 0.01 --exp-name lr_01

# Compare in TensorBoard
tensorboard --logdir ./runs
```

Expected improvements:
- Early learning rate too high: unstable
- Good learning rate: smooth convergence
- Very low learning rate: slow convergence

#### Deep: Grid/Random Hyperparameter Search

**Grid Search (12 trials):**
```bash
python hyperparameter_search.py --method grid --num-trials 12
```

**Random Search (20 trials):**
```bash
python hyperparameter_search.py --method random --num-trials 20
```

**Output:**
```
HYPERPARAMETER SEARCH RESULTS (GRID)

Trial | Learning Rate | Batch Size | Optimizer | Weight Decay | Test Accuracy
------|---------------|-----------|-----------|-------------|---------------
1     | 0.001000      | 64        | Adam      | 0.0000      | 98.50%
2     | 0.001000      | 128       | Adam      | 0.0000      | 97.80%
...
```

Produces:
- Results table saved to `runs/search_*/results_table.txt`
- JSON results for parsing
- TensorBoard logs for each trial

---

### Direction #3: Architecture Engineering

#### Basic: Compare Architectures

```bash
python architecture_explorer.py --exp-name arch_comparison
```

Compares:
- SimpleCNN (baseline)
- SimpleCNNWithBatchNorm (adds normalization)
- DeepCNN (deeper network)
- SimpleCNNWithLeakyReLU (different activation)
- SimpleCNNWithELU (different activation)

Output:
```
ARCHITECTURE COMPARISON RESULTS

architecture              | parameters | train_loss | val_loss | val_accuracy | test_accuracy
--------------------------|------------|-----------|----------|--------------|---------------
SimpleCNNWithBatchNorm   | 25,000     | 0.0234    | 0.0512   | 98.50%      | 98.40%
SimpleCNNWithLeakyReLU   | 25,000     | 0.0245    | 0.0498   | 98.35%      | 98.20%
...
```

#### Deep: Architecture Variants

Create new architectures in `model.py`:

```python
class MyCustomCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # Add custom layers
    
    def forward(self, x):
        # Custom forward pass
        return x
```

Then compare with:
```bash
python architecture_explorer.py
```

---

### Direction #4: Training Acceleration & Profiling

#### Basic: Batch Size Effects

```bash
# Small batch size
python main.py --batch-size 32 --exp-name bs_32

# Medium batch size
python main.py --batch-size 64 --exp-name bs_64

# Large batch size
python main.py --batch-size 256 --exp-name bs_256
```

Monitor:
- Training time per epoch
- GPU memory usage (nvidia-smi)
- Convergence speed

#### Deep: Systematic Profiling

```bash
python hyperparameter_search.py --method random --num-trials 15
```

This explores:
- Different batch sizes: 16, 32, 64, 128, 256
- Different learning rates
- Optimizer choices

Results show:
- Time per epoch
- Memory efficiency
- Convergence behavior

---

### Direction #5: Reproducibility & Determinism

#### Basic: Check Reproducibility

```bash
# Run 1
python main.py --seed 42 --exp-name run1

# Run 2
python main.py --seed 42 --exp-name run2

# Run 3 (different seed)
python main.py --seed 123 --exp-name run3
```

Compare in TensorBoard:
- Loss curves should be identical for same seed
- Different with different seed

#### Deep: Variability Analysis

```bash
# Multiple runs with same seed
for i in 1 2 3; do
  python main.py --seed 42 --exp-name determinism_run$i
done

# Multiple runs with different seeds
for seed in 42 100 200; do
  python main.py --seed $seed --exp-name seed_$seed
done
```

Analysis:
- Identical results with same seed → reproducible
- Differences with different seeds → normal
- Within-run variability → GPU non-determinism level

---

### Direction #6: Observability & TensorBoard

#### Basic: Standard Logging

Automatically logged by `main.py`:
- Training loss per batch
- Validation loss per epoch
- Validation accuracy per epoch
- Learning rate
- Sample input images
- Weight histograms

```bash
python main.py
tensorboard --logdir ./runs
```

View:
- SCALARS tab: loss, accuracy curves
- IMAGES tab: sample MNIST digits
- HISTOGRAMS tab: weight distributions

#### Deep: Advanced Logging

See `train.py` for customization options:

```python
# Log custom metrics
writer.add_scalar("custom_metric", value, epoch)

# Log feature maps (see commented code)
feature_maps = activations["conv1"]
grid = vutils.make_grid(feature_maps)
writer.add_image("FeatureMaps/conv1", grid, epoch)
```

Uncomment to enable advanced logging in `train.py`.

---

### Direction #7: Data Engineering & Augmentation

#### Basic: Apply Augmentation

Edit `config.yaml`:
```yaml
data:
  augmentation: true
```

Or modify `preprocessing.py`:
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

#### Deep: Augmentation Analysis

```bash
# Without augmentation
python main.py --exp-name no_augmentation

# With augmentation
python main.py --exp-name with_augmentation
```

Compare:
- Convergence speed
- Final accuracy
- Generalization to test set
- Robustness

---

### Direction #8: Extended Inference / OOD Testing

#### Basic: Evaluation on Test Set

```bash
python main.py
# Automatically evaluates on test set
# Prints: "Test Accuracy: XX.XX%"
```

#### Deep: Custom Inference

```python
import torch
from model import get_model
from preprocessing import load_mnist_data

# Load model
model = get_model("SimpleCNN", device="cuda")
model.load_state_dict(torch.load("runs/exp_.../best_model.pt")["model_state_dict"])

# Evaluate
_, test_loader = load_mnist_data()
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.cuda())
        confidences = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        # Analyze OOD behavior
```

---

### Direction #9: Transfer Learning

#### Basic: Pre-trained ResNet-18

Add to `model.py`:
```python
import torchvision.models as models

class ResNet18MNIST(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        return self.backbone(x)
```

Then use:
```bash
python main.py --model ResNet18MNIST
```

#### Deep: Fine-tuning Strategies

```python
# Strategy 1: Frozen backbone
for name, param in model.backbone.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# Strategy 2: Partial fine-tuning
for name, param in model.backbone.named_parameters():
    if "layer3" not in name and "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# Strategy 3: Full fine-tuning
# Keep all requires_grad = True
```

---

## Complete Workflow Example

### Scenario: Optimize MNIST Classification

```bash
# 1. Enter environment
cd /home/michal/AGH/sem6/DL_CUDA
nix-shell shell.nix
cd dlcuda/project

# 2. Test setup
bash test_setup.sh

# 3. Baseline training
python main.py --exp-name baseline_simple --epochs 10 --save-plots

# 4. Try different learning rates (Direction #2 Basic)
python main.py --learning-rate 0.0005 --exp-name lr_tuning_0005
python main.py --learning-rate 0.002 --exp-name lr_tuning_002

# 5. Compare architectures (Direction #3 Basic)
python architecture_explorer.py --exp-name arch_exp1

# 6. Hyperparameter grid search (Deep Direction #2)
python hyperparameter_search.py --method grid --num-trials 15 --exp-name grid_search_1

# 7. Monitor in TensorBoard (new terminal)
tensorboard --logdir ./runs

# 8. Analyze results
# - View all experiment curves in TensorBoard
# - Check results_table.txt files
# - Find best configuration
# - Fine-tune around best parameters

# 9. Final training with best params
python main.py \
  --learning-rate 0.001 \
  --batch-size 64 \
  --optimizer Adam \
  --weight-decay 0.0001 \
  --epochs 20 \
  --model SimpleCNNWithBatchNorm \
  --exp-name final_best_model \
  --save-plots
```

---

## File Description

### Core Files

**`main.py`** - Entry point
- Argument parsing
- Configuration management
- Training pipeline orchestration
- Results logging

**`preprocessing.py`** - Data handling
- MNIST loading
- Transforms and augmentation
- DataLoader creation
- Dataset utilities

**`model.py`** - Model definitions
- SimpleCNN (baseline)
- SimpleCNNWithBatchNorm
- DeepCNN
- SimpleCNNWithLeakyReLU
- SimpleCNNWithELU
- Utility functions for model creation

**`train.py`** - Training logic
- Trainer class
- Training loop
- Validation
- Checkpointing
- TensorBoard logging
- Convenience function: train_model()

**`utils.py`** - Utilities
- Reproducibility (seeds, determinism)
- Device management
- Evaluation metrics
- Plotting and visualization
- Configuration I/O
- Results table formatting

### Exploration Files

**`hyperparameter_search.py`** - Deep Direction #2
- Grid search over hyperparameters
- Random search
- Multi-trial experiment execution
- Results aggregation and reporting

**`architecture_explorer.py`** - Direction #3
- Compare multiple architectures
- Run parallel training
- Generate comparison tables
- Analyze architecture effects

### Configuration

**`config.yaml`** - YAML configuration
- Training parameters
- Data settings
- Model selection
- Device and logging

---

## Expected Results After Following This Guide

### After Basic Exploration:
✓ Understand full deep learning pipeline
✓ Run training with different hyperparameters
✓ Compare multiple architectures
✓ Monitor training with TensorBoard
✓ Save and analyze results

### After Deep Exploration (Directions #1 & #2):
✓ Run 10-20+ controlled hyperparameter experiments
✓ Generate results tables with rankings
✓ Identify best configurations
✓ Full reproducibility across runs
✓ Professional experiment tracking

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Make sure you're in nix-shell
```bash
nix-shell shell.nix  # Enter environment first
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
python main.py --batch-size 32
```

### Issue: "TensorBoard shows no data"
**Solution:** 
1. Kill old TensorBoard: `pkill tensorboard`
2. Restart: `tensorboard --logdir ./runs`
3. Wait 5 seconds for data to load
4. Refresh browser

### Issue: Training is very slow
**Solution:** Check GPU usage
```bash
nvidia-smi  # Should show GPU memory being used
python main.py --batch-size 256  # Larger batches
```

### Issue: Results not saved
**Solution:** Check disk space and permissions
```bash
ls -la ./runs/  # Verify directory exists
df -h  # Check disk space
```

---

## Next Steps

1. ✓ Complete all basic directions
2. ✓ Run Deep Direction #1 and #2
3. ? Try creating custom architectures
4. ? Implement data augmentation variants
5. ? Add more advanced monitoring
6. ? Deploy model for inference

---

## References

- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **CNN Architectures**: https://arxiv.org/abs/1512.03385 (ResNet)

Good luck! 🚀
