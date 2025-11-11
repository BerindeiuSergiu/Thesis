#!/bin/bash
# Segmentation Models Installation Script

echo "Installing segmentation models for thesis project..."

# Basic requirements (should already be installed)
pip install torch torchvision opencv-python pillow numpy

# Detectron2 (adjust CUDA version as needed)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM model weights
mkdir -p ../data/models
cd ../data/models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../../notebooks

echo "Installation complete! Run the notebook again to test all methods."
