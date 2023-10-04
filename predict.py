from diffusers import VersatileDiffusionImageVariationPipeline
import torch
from cog import BasePredictor, Input, Path
import os
import cv2
import shutil
import requests
import numpy as np
from PIL import Image
from io import BytesIO

VERSATILE_MODEL = "shi-labs/versatile-diffusion"
VERSATILE_CACHE = "versatile-cache"

class Predictor(BasePredictor):
  
  def setup(self):
      
    """Load the model"""
    t1 = time.time()
    print("Loading versatile diffusion model")
    versatile = VersatileDiffusionImageVariationPipeline.from_pretrained(
        VERSATILE_CACHE,
        torch_dtype=torch.float16
    )
    self.pipe = pipe.to("cuda")
    t2 = time.time()
    print("Setup took: ", t2 - t1)

  def load_image(self, path):
    shutil.copyfile(path, "/tmp/image.png")
    return load_image("/tmp/image.png").convert("RGB")

  @torch.inference_model()
  def predict(
    self,
    image: Path = Input(
      description="Input image for image variation",
      default=None,
    ),
    seed: int = Input(
      description="Random seed. Set to 0 to randomize the seed", default=0
    ),
  ) -> Path:
      """Run a single prediction on the model"""
      if (seed is None) or (seed <= 0):
          seed = int.from_bytes(os.urandom(2), "big")
      generator = torch.Generator("cuda").manual_seed(seed)
      print(f"Using seed: {seed}")
    
      image = self.load_image(image)
      image = Image.open(image)
      image = self.pipe(image, generator=generator).images
    
      output_path = f"/tmp/output.png"
      images[0].save(output_path)

      return Path(output_path)
