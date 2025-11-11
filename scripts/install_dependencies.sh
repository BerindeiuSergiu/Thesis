#!/bin/bash
# Installation script for Thesis Project Dependencies
# This script installs all required packages for the 3D Room Reconstruction project

echo "Installing Thesis Project Dependencies"
echo "========================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment detected!"
    echo "   It's recommended to run this in a virtual environment."
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "Step 2: Installing core ML and computer vision packages..."
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0
pip install numpy>=1.24.0
pip install opencv-python>=4.8.0
pip install pillow>=9.5.0

echo ""
echo "Step 3: Installing data science and visualization packages..."
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install plotly>=5.14.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0
pip install scikit-image>=0.20.0

echo ""
echo "Step 4: Installing video processing packages..."
pip install imageio>=2.31.0
pip install imageio-ffmpeg>=0.4.8
pip install moviepy>=1.0.3
pip install ffmpeg-python>=0.2.0

echo ""
echo "Step 5: Installing 3D processing packages..."
pip install open3d>=0.17.0
pip install trimesh>=3.21.0

echo ""
echo "Step 6: Installing deep learning models and transformers..."
pip install transformers>=4.30.0
pip install timm>=0.9.0

echo ""
echo "Step 7: Installing development tools..."
pip install jupyter>=1.0.0
pip install jupyterlab>=4.0.0
pip install tqdm>=4.65.0
pip install ipywidgets>=8.0.0

echo ""
echo "Step 8: Installing configuration and utility packages..."
pip install pyyaml>=6.0
pip install python-dotenv>=1.0.0
pip install click>=8.1.0
pip install rich>=13.4.0

echo ""
echo "Step 9: Installing testing packages..."
pip install pytest>=7.4.0
pip install pytest-cov>=4.1.0

echo ""
echo "Step 10: Installing optional advanced packages..."
echo "   (These may fail if dependencies are not available - that's OK)"

# Try to install segment-anything
echo "   Attempting to install Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git || echo "   WARNING: Segment Anything installation failed (optional)"

# Try to install detectron2 based on system
echo "   Attempting to install Detectron2..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU detected, installing CUDA version..."
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html || echo "   WARNING: Detectron2 CUDA installation failed (optional)"
else
    echo "   No GPU detected, installing CPU version..."
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html || echo "   WARNING: Detectron2 CPU installation failed (optional)"
fi

echo ""
echo "Step 11: Testing installation..."
python -c "
import sys
import importlib

# Test core packages
packages_to_test = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
    'cv2', 'PIL', 'torch', 'torchvision', 'sklearn',
    'imageio', 'moviepy.editor', 'open3d', 'tqdm',
    'yaml', 'pathlib', 'jupyter'
]

failed_packages = []
for package in packages_to_test:
    try:
        if '.' in package:
            # Handle submodules like moviepy.editor
            main_package = package.split('.')[0]
            importlib.import_module(main_package)
        else:
            importlib.import_module(package)
        print(f'SUCCESS: {package}')
    except ImportError as e:
        print(f'FAILED: {package}: {e}')
        failed_packages.append(package)

if failed_packages:
    print(f'\\nWARNING: {len(failed_packages)} packages failed to import: {failed_packages}')
    print('   You may need to install them manually.')
else:
    print('\\nAll core packages installed successfully!')
"

echo ""
echo "Installation Summary"
echo "======================"
echo "Core ML packages (PyTorch, NumPy, OpenCV)"
echo "Data science packages (Pandas, Matplotlib, Scikit-learn)"  
echo "Video processing packages (ImageIO, MoviePy)"
echo "3D processing packages (Open3D)"
echo "Development tools (Jupyter, tqdm)"
echo "Utility packages (YAML, Rich)"
echo ""
echo "Optional packages (may need manual installation):"
echo "   - Segment Anything (for state-of-the-art segmentation)"
echo "   - Detectron2 (for Mask R-CNN)"
echo "   - COLMAP (for traditional SfM - install separately)"
echo ""
echo "Next steps:"
echo "   1. Run: jupyter lab"
echo "   2. Open: experiments/notebooks/01_video_segmentation_comparison.ipynb"
echo "   3. Test your installation by running the notebook"
echo ""
echo "Installation complete!"