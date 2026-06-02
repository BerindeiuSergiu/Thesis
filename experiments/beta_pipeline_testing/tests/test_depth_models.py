"""Test script to compare MiDaS vs Marigold on a single frame."""
import cv2
import numpy as np
from pathlib import Path

from pipelines.depth_legacy.config import DepthEstimationConfig
from pipelines.depth_legacy.depth_estimator import create_depth_estimator
from pipelines.shared.paths import OUTPUTS_ROOT

def test_depth_models():
    """Test and compare both depth models."""
    
    # Load a test frame
    frames_dir = OUTPUTS_ROOT / "legacy_depth" / "frames"
    if not frames_dir.exists():
        print("ERROR: No frames found. Run main.py first.")
        return
    
    test_frame_path = list(frames_dir.glob("*.jpg"))[0]
    frame = cv2.imread(str(test_frame_path))
    
    print(f"Testing with frame: {test_frame_path.name}")
    print(f"Frame shape: {frame.shape}\n")
    
    # Test 1: MiDaS
    print("=" * 70)
    print("MODEL 1: MiDaS DPT_Hybrid")
    print("=" * 70)
    try:
        config_midas = DepthEstimationConfig(
            model_type="midas",
            depth_model="dpt_hybrid",
            device="cuda"
        )
        estimator_midas = create_depth_estimator(config_midas)
        depth_midas = estimator_midas.estimate_depth(frame)
        
        print(f"✓ Depth map shape: {depth_midas.shape}")
        print(f"✓ Depth range: [{depth_midas.min():.4f}, {depth_midas.max():.4f}]")
        print(f"✓ Data type: {depth_midas.dtype}")
        print(f"✓ Characteristics: Relative depth (normalized 0-1)")
        
    except Exception as e:
        print(f"✗ MiDaS failed: {e}")
    
    print()
    
    # Test 2: Marigold
    print("=" * 70)
    print("MODEL 2: Marigold (Metric Depth)")
    print("=" * 70)
    try:
        config_marigold = DepthEstimationConfig(
            model_type="marigold",
            marigold_model="Bing-su/marigold-depth-v2-0",
            half_precision=True,
            device="cuda"
        )
        estimator_marigold = create_depth_estimator(config_marigold)
        depth_marigold = estimator_marigold.estimate_depth(frame)
        
        print(f"✓ Depth map shape: {depth_marigold.shape}")
        print(f"✓ Depth range: [{depth_marigold.min():.4f}, {depth_marigold.max():.4f}]")
        print(f"✓ Data type: {depth_marigold.dtype}")
        print(f"✓ Characteristics: Metric depth (absolute scale)")
        
    except Exception as e:
        print(f"✗ Marigold failed: {e}")
        print("  Install: pip install diffusers")
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("""
MiDaS (Relative Depth):
  - Pros: Fast, good for quick tests
  - Cons: No absolute scale, noisier
  - Use: Prototyping, visualization

Marigold (Metric Depth):
  - Pros: Absolute scale, better quality, excellent for fusion
  - Cons: Slower, higher memory
  - Use: Production, SfM fusion, best results

RECOMMENDATION: Use Marigold for your thesis work!
""")

if __name__ == "__main__":
    test_depth_models()
