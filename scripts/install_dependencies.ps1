# PowerShell script for installing Thesis dependencies
# Run this script in PowerShell as Administrator if needed

Write-Host "Installing Thesis Project Dependencies" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check if we're in a virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "Warning: No virtual environment detected!" -ForegroundColor Yellow
    Write-Host "   It's recommended to run this in a virtual environment." -ForegroundColor Yellow
    $continue = Read-Host "   Continue anyway? (y/N)"
    if ($continue -ne "y") {
        exit 1
    }
}

Write-Host ""
Write-Host "Step 1: Upgrading pip and installing build tools..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host ""
Write-Host "Step 2: Installing from requirements.txt..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "WARNING: requirements.txt not found, installing core packages manually..." -ForegroundColor Yellow
    
    # Core packages
    $packages = @(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "opencv-python>=4.8.0",
        "pillow>=9.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.20.0",
        "imageio>=2.31.0",
        "moviepy>=1.0.3",
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0"
    )
    
    foreach ($package in $packages) {
        Write-Host "   Installing $package..." -ForegroundColor Gray
        pip install $package
    }
}

Write-Host ""
Write-Host "Step 3: Installing optional packages..." -ForegroundColor Cyan
Write-Host "   (These may fail - that's OK)" -ForegroundColor Gray

# Try to install optional packages
$optionalPackages = @(
    "open3d>=0.17.0",
    "trimesh>=3.21.0"
)

foreach ($package in $optionalPackages) {
    try {
        Write-Host "   Installing $package..." -ForegroundColor Gray
        pip install $package
        Write-Host "   SUCCESS: $package" -ForegroundColor Green
    } catch {
        Write-Host "   WARNING: $package failed (optional)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Testing installation..." -ForegroundColor Cyan

# Test Python imports
$testScript = @"
import sys
packages_to_test = ['numpy', 'pandas', 'matplotlib', 'cv2', 'PIL', 'torch', 'sklearn', 'plotly', 'imageio', 'tqdm', 'yaml', 'jupyter']
failed = []
for package in packages_to_test:
    try:
        __import__(package)
        print(f'SUCCESS: {package}')
    except ImportError:
        print(f'FAILED: {package}')
        failed.append(package)
if failed:
    print(f'\\nWARNING: Failed imports: {failed}')
else:
    print('\\nAll core packages imported successfully!')
"@

python -c $testScript

Write-Host ""
Write-Host "Installation Summary" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green
Write-Host "Core packages installed" -ForegroundColor Green
Write-Host "Development tools ready" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "   1. Run: jupyter lab" -ForegroundColor White
Write-Host "   2. Open: experiments/notebooks/01_video_segmentation_comparison.ipynb" -ForegroundColor White
Write-Host "   3. Test your installation" -ForegroundColor White
Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green

# Keep PowerShell window open
Read-Host "Press Enter to continue..."