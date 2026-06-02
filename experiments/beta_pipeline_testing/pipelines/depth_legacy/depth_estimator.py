"""Depth estimation module supporting MiDaS and Marigold."""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from .config import DepthEstimationConfig


class DepthEstimatorBase:
    """Base class for depth estimators."""
    
    def __init__(self, config: DepthEstimationConfig):
        """Initialize depth estimator.
        
        Args:
            config: DepthEstimationConfig with model settings
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single frame.
        
        Args:
            frame: Input frame (BGR, numpy array)
            
        Returns:
            Normalized depth map (0-1)
        """
        raise NotImplementedError
    
    def estimate_batch(self, frames: list) -> list:
        """Estimate depth for multiple frames.
        
        Args:
            frames: List of frames (numpy arrays)
            
        Returns:
            List of normalized depth maps
        """
        depth_maps = []
        for i, frame in enumerate(frames):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(frames)} frames")
            depth = self.estimate_depth(frame)
            depth_maps.append(depth)
        
        # Final cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return depth_maps
    
    @staticmethod
    def _mock_depth(frame: np.ndarray) -> np.ndarray:
        """Create synthetic depth map (fallback)."""
        height, width = frame.shape[:2]
        y = np.linspace(0, 1, height)
        x = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)
        depth = (X + Y) / 2.0
        depth += 0.1 * np.sin(X * 10) * np.cos(Y * 10)
        return depth.astype(np.float32)


class MiDaSDepthEstimator(DepthEstimatorBase):
    """MiDaS v3.1 depth estimation."""
    
    def __init__(self, config: DepthEstimationConfig):
        """Initialize MiDaS depth estimator."""
        super().__init__(config)
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model from torch hub."""
        print(f"Loading MiDaS model: {self.config.depth_model}...")
        
        try:
            # Load model from torch hub using legacy names
            model_name = self.config.depth_model
            
            # Map friendly names to torch hub names
            model_map = {
                "dpt_large": "DPT_Large",
                "dpt_hybrid": "DPT_Hybrid",
                "midas_v21_small": "MiDaS_small"
            }
            
            hub_name = model_map.get(model_name, "DPT_Hybrid")
            
            # Load model from torch hub
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                hub_name
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms"
            )
            
            # Select transform based on model
            if "dpt" in model_name.lower():
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            print(f"MiDaS model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading MiDaS: {e}")
            self.model = None
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single frame using MiDaS."""
        if self.model is None:
            return self._mock_depth(frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            input_batch = self.transform(frame_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Interpolate on GPU
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                # Normalize
                depth_min = prediction.min()
                depth_max = prediction.max()
                depth = (prediction - depth_min) / (depth_max - depth_min + 1e-8)
                
                # Transfer to CPU only at the end
                depth = depth.cpu().numpy()
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return depth.astype(np.float32)
            
        except Exception as e:
            print(f"MiDaS inference error: {e}")
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return self._mock_depth(frame)


class MarigoldDepthEstimator(DepthEstimatorBase):
    """Marigold metric depth estimation (diffusers pipeline)."""
    
    def __init__(self, config: DepthEstimationConfig):
        """Initialize Marigold depth estimator."""
        super().__init__(config)
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load Marigold pipeline."""
        print(f"Loading Marigold model: {self.config.marigold_model}...")
        
        try:
            from diffusers import MarigoldDepthPipeline
        except ImportError:
            print("ERROR: diffusers library required for Marigold")
            print("Install: pip install diffusers")
            self.pipeline = None
            return
        
        try:
            # Load pipeline (let it auto-detect dtype)
            self.pipeline = MarigoldDepthPipeline.from_pretrained(
                self.config.marigold_model
            )
            self.pipeline = self.pipeline.to(self.device)
            
            print(f"Marigold model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Marigold: {e}")
            self.pipeline = None
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single frame using Marigold."""
        if self.pipeline is None:
            return self._mock_depth(frame)
        
        try:
            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Inference - Marigold returns a dataclass with 'prediction' attribute
            with torch.no_grad():
                output = self.pipeline(pil_image)
            
            # Extract depth map from output.prediction
            # prediction is shape (1, H, W, 1) as numpy array
            depth = output.prediction[0, :, :, 0]  # Extract to (H, W)
            
            # Normalize to 0-1 range for consistency with other models
            depth_min = depth.min()
            depth_max = depth.max()
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return depth.astype(np.float32)
            
        except Exception as e:
            print(f"Marigold inference error: {e}")
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return self._mock_depth(frame)


def create_depth_estimator(config: DepthEstimationConfig) -> DepthEstimatorBase:
    """Factory function to create appropriate depth estimator.
    
    Args:
        config: DepthEstimationConfig with model_type specified
        
    Returns:
        Appropriate depth estimator instance
    """
    if config.model_type.lower() == "marigold":
        print("\n[DEPTH] Using Marigold metric depth estimation")
        return MarigoldDepthEstimator(config)
    else:
        print("\n[DEPTH] Using MiDaS depth estimation")
        return MiDaSDepthEstimator(config)

