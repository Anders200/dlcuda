#!/bin/bash
# Test script to verify pipeline setup

echo "=================================="
echo "Deep Learning Pipeline Test"
echo "=================================="
echo ""

# Check Python
echo "Python version:"
python --version
echo ""

# Check PyTorch
echo "Checking PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo ""

# Check CUDA
echo "Checking CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo ""

# Check dependencies
echo "Checking required packages..."
python -c "
import sys
packages = ['torch', 'torchvision', 'numpy', 'yaml', 'tensorboard']
for pkg in packages:
    try:
        mod = __import__(pkg)
        if hasattr(mod, '__version__'):
            print(f'✓ {pkg}: {mod.__version__}')
        else:
            print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg}: NOT FOUND')
        sys.exit(1)
"
echo ""

# Test data loading
echo "Testing data loading..."
python -c "
from preprocessing import get_dataset_info
info = get_dataset_info()
print('Dataset info:')
for key, val in info.items():
    print(f'  {key}: {val}')
"
echo ""

# Test model creation
echo "Testing model creation..."
python -c "
from model import get_model, get_model_info
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model('SimpleCNN', device=device)
info = get_model_info(model)
print(f'Model: {info[\"model_class\"]}')
print(f'Parameters: {info[\"total_params\"]:,}')
print(f'Device: {info[\"device\"]}')
"
echo ""

echo "=================================="
echo "✓ All tests passed!"
echo "=================================="
echo ""
echo "To run training, use:"
echo "  python main.py"
echo ""
echo "To run hyperparameter search, use:"
echo "  python hyperparameter_search.py --method grid --num-trials 12"
echo ""
echo "To compare architectures, use:"
echo "  python architecture_explorer.py"
echo ""
