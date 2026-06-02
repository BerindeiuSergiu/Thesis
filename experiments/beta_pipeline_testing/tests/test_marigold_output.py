"""Test Marigold output access."""
from diffusers import MarigoldDepthPipeline
from PIL import Image
import torch
import numpy as np

print("Testing Marigold output access...")
pipeline = MarigoldDepthPipeline.from_pretrained(
    'prs-eth/marigold-depth-v1-0',
    dtype=torch.float16,
    variant="fp16"
).to('cuda')

dummy_img = Image.new('RGB', (512, 384))
with torch.no_grad():
    output = pipeline(dummy_img)

print(f"Output type: {type(output)}")
print(f"Has depth_pt: {hasattr(output, 'depth_pt')}")
if hasattr(output, 'depth_pt'):
    depth_tensor = output.depth_pt
    depth_np = depth_tensor.cpu().numpy()
    print(f"Depth shape: {depth_np.shape}")
    print(f"Depth dtype: {depth_np.dtype}")
    print(f"Depth range: {depth_np.min():.4f} - {depth_np.max():.4f}")
    print("SUCCESS: Marigold output access fixed!")
