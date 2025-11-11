# Dependency Installation Scripts

This folder contains scripts to install all required dependencies for the Thesis project.

## Quick Start (Windows with venv)

1. **Activate your virtual environment:**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Run the Python installer:**
   ```powershell
   python scripts/install_dependencies.py
   ```

## Available Installation Methods

### 1. Python Script (Recommended)
```bash
python scripts/install_dependencies.py
```
- Cross-platform (Windows/Linux/Mac)
- Automatically detects virtual environment
- Tests all imports after installation
- Provides detailed feedback

### 2. Requirements File
```bash
pip install -r requirements.txt
```
- Standard pip installation
- All versions specified
- May need manual handling of optional packages

### 3. Platform-Specific Scripts

#### Windows (PowerShell)
```powershell
.\scripts\install_dependencies.ps1
```

#### Windows (Batch)
```cmd
scripts\install_dependencies.bat
```

#### Linux/Mac (Bash)
```bash
bash scripts/install_dependencies.sh
```

## Virtual Environment Setup

If you don't have a virtual environment yet:

### Windows
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux/Mac
```bash
python -m venv .venv
source .venv/bin/activate
```

## What Gets Installed

### Core Packages
- **Data Science**: numpy, pandas, matplotlib, seaborn, plotly
- **Machine Learning**: torch, torchvision, scikit-learn, scikit-image
- **Computer Vision**: opencv-python, pillow
- **Video Processing**: imageio, moviepy, ffmpeg-python

### Development Tools
- **Jupyter**: jupyter, jupyterlab, ipywidgets
- **Utilities**: tqdm, pyyaml, rich, click

### Optional Packages (may require manual installation)
- **Segment Anything**: State-of-the-art segmentation
- **Detectron2**: Mask R-CNN implementation
- **Open3D**: 3D processing and visualization
- **COLMAP**: Traditional structure-from-motion

## Troubleshooting

### Common Issues

1. **PyTorch GPU Support**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Detectron2 Installation**
   ```bash
   # CUDA version
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
   # CPU version  
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
   ```

3. **Open3D Issues**
   ```bash
   pip install --upgrade pip
   pip install open3d --no-cache-dir
   ```

4. **FFmpeg Not Found**
   - Windows: Download from https://ffmpeg.org/
   - Ubuntu: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

### Memory Issues
If you encounter memory issues during installation:
```bash
pip install --no-cache-dir -r requirements.txt
```

## Testing Your Installation

After installation, test that everything works:

1. **Run the test script:**
   ```python
   python -c "
   import numpy, pandas, matplotlib, cv2, torch, sklearn
   print('All core packages imported successfully!')
   "
   ```

2. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

3. **Open and run the first notebook:**
   `experiments/notebooks/01_video_segmentation_comparison.ipynb`

## System Requirements

- **Python**: 3.8+ (3.12 recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space for packages
- **GPU**: Optional but recommended for deep learning models

## Getting Help

If you encounter issues:

1. Check the installation logs for specific error messages
2. Ensure you're in the virtual environment
3. Try installing packages individually to isolate issues
4. Check the package documentation for system-specific requirements

## Next Steps

After successful installation:
1. Run the video segmentation comparison notebook
2. Test frame extraction methods
3. Begin developing your 3D reconstruction pipeline

---

**Note**: The installation scripts are designed to be safe and will not overwrite existing packages unless explicitly upgrading. Always use a virtual environment to avoid conflicts with system packages.