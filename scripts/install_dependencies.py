#!/usr/bin/env python3
"""
Thesis Project Dependency Installer
Automatically installs all required packages for the 3D Room Reconstruction project
Works with virtual environments and provides detailed feedback
"""

import subprocess
import sys
import importlib
from pathlib import Path

class ThesisInstaller:
    def __init__(self):
        self.failed_packages = []
        self.successful_packages = []
        
        # Core packages required by the notebooks
        self.core_packages = [
            "numpy>=1.24.0",
            "pandas>=2.0.0", 
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "opencv-python>=4.8.0",
            "pillow>=9.5.0",
            "scikit-learn>=1.3.0",
            "scikit-image>=0.20.0",
            "scipy>=1.10.0"
        ]
        
        self.ml_packages = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "torchaudio>=2.0.0",
            "transformers>=4.30.0",
            "timm>=0.9.0"
        ]
        
        self.video_packages = [
            "imageio>=2.31.0",
            "imageio-ffmpeg>=0.4.8",
            "moviepy>=1.0.3",
            "ffmpeg-python>=0.2.0"
        ]
        
        self.dev_packages = [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
            "tqdm>=4.65.0",
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0",
            "click>=8.1.0",
            "rich>=13.4.0",
            "pytest>=7.4.0"
        ]
        
        self.optional_packages = [
            "open3d>=0.17.0",
            "trimesh>=3.21.0"
        ]
        
    def check_venv(self):
        """Check if we're in a virtual environment"""
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv:
            print("Virtual environment detected")
            print(f"   Python: {sys.executable}")
            print(f"   Virtual env: {sys.prefix}")
            return True
        else:
            print("Warning: Not in a virtual environment!")
            print("   It's recommended to run this in a virtual environment.")
            response = input("   Continue anyway? (y/N): ").lower().strip()
            return response == 'y'
    
    def upgrade_pip(self):
        """Upgrade pip and essential tools"""
        print("\nUpgrading pip and build tools...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
            print("Pip upgrade successful")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Pip upgrade failed: {e}")
            return False
    
    def install_packages(self, packages, category_name):
        """Install a list of packages"""
        print(f"\nInstalling {category_name}...")
        
        for package in packages:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                self.successful_packages.append(package)
                print(f"   SUCCESS: {package}")
            except subprocess.CalledProcessError:
                print(f"   FAILED: {package}")
                self.failed_packages.append(package)
    
    def install_optional_packages(self):
        """Install optional packages that might fail"""
        print(f"\nInstalling optional advanced packages...")
        print("   (These may fail if dependencies are not available - that's OK)")
        
        # Try to install segment-anything
        print("   Attempting to install Segment Anything...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                 "git+https://github.com/facebookresearch/segment-anything.git"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("   SUCCESS: Segment Anything")
            self.successful_packages.append("segment-anything")
        except subprocess.CalledProcessError:
            print("   WARNING: Segment Anything installation failed (optional)")
            
        # Try to install detectron2
        print("   Attempting to install Detectron2...")
        try:
            # Check if CUDA is available
            try:
                subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
                cuda_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"
                print("   GPU detected, installing CUDA version...")
            except (subprocess.CalledProcessError, FileNotFoundError):
                cuda_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html"
                print("   No GPU detected, installing CPU version...")
                
            subprocess.check_call([sys.executable, "-m", "pip", "install", "detectron2", "-f", cuda_url],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("   SUCCESS: Detectron2")
            self.successful_packages.append("detectron2")
        except subprocess.CalledProcessError:
            print("   WARNING: Detectron2 installation failed (optional)")
    
    def test_imports(self):
        """Test that key packages can be imported"""
        print("\nTesting package imports...")
        
        test_packages = {
            'numpy': 'numpy',
            'pandas': 'pandas', 
            'matplotlib': 'matplotlib.pyplot',
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'torch': 'torch',
            'sklearn': 'scikit-learn',
            'plotly': 'plotly',
            'imageio': 'imageio',
            'moviepy.editor': 'moviepy',
            'tqdm': 'tqdm',
            'yaml': 'pyyaml',
            'jupyter': 'jupyter'
        }
        
        import_failures = []
        
        for import_name, package_name in test_packages.items():
            try:
                if '.' in import_name:
                    # Handle submodules like moviepy.editor
                    main_package = import_name.split('.')[0]
                    importlib.import_module(main_package)
                else:
                    importlib.import_module(import_name)
                print(f"   SUCCESS: {import_name}")
            except ImportError as e:
                print(f"   FAILED: {import_name}: {e}")
                import_failures.append(package_name)
        
        return import_failures
    
    def print_summary(self, import_failures):
        """Print installation summary"""
        print("\nInstallation Summary")
        print("=" * 50)
        
        if self.successful_packages:
            print(f"SUCCESS: Successfully installed {len(self.successful_packages)} packages:")
            for pkg in self.successful_packages[:10]:  # Show first 10
                print(f"   - {pkg}")
            if len(self.successful_packages) > 10:
                print(f"   ... and {len(self.successful_packages) - 10} more")
        
        if self.failed_packages:
            print(f"\nFAILED: Failed to install {len(self.failed_packages)} packages:")
            for pkg in self.failed_packages:
                print(f"   - {pkg}")
        
        if import_failures:
            print(f"\nWARNING: {len(import_failures)} packages failed import test:")
            for pkg in import_failures:
                print(f"   - {pkg}")
        
        print("\nManual installation may be needed for:")
        print("   - COLMAP (for traditional Structure-from-Motion)")
        print("   - FFmpeg (for advanced video processing)")
        print("   - CUDA toolkit (for GPU acceleration)")
        
        print("\nNext steps:")
        print("   1. Activate your virtual environment (if not already active)")
        print("   2. Run: jupyter lab")
        print("   3. Open: experiments/notebooks/01_video_segmentation_comparison.ipynb")
        print("   4. Test your installation by running the first notebook")
    
    def install_all(self):
        """Main installation process"""
        print("Thesis Project Dependency Installer")
        print("=" * 50)
        print("Installing packages for 3D Room Reconstruction project")
        
        # Check virtual environment
        if not self.check_venv():
            print("Installation cancelled.")
            return False
        
        # Upgrade pip
        if not self.upgrade_pip():
            print("WARNING: Pip upgrade failed, continuing anyway...")
        
        # Install package groups
        self.install_packages(self.core_packages, "core data science packages")
        self.install_packages(self.ml_packages, "machine learning packages")  
        self.install_packages(self.video_packages, "video processing packages")
        self.install_packages(self.dev_packages, "development tools")
        self.install_packages(self.optional_packages, "3D processing packages")
        
        # Install optional packages
        self.install_optional_packages()
        
        # Test imports
        import_failures = self.test_imports()
        
        # Print summary
        self.print_summary(import_failures)
        
        print("\nInstallation process complete!")
        
        return len(self.failed_packages) == 0 and len(import_failures) == 0

if __name__ == "__main__":
    installer = ThesisInstaller()
    success = installer.install_all()
    
    if success:
        print("\nAll packages installed successfully!")
        sys.exit(0)
    else:
        print("\nSome packages failed to install. Check the summary above.")
        sys.exit(1)