import cv2
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms
import tqdm
from collections import OrderedDict
from diffusers.models.autoencoders.vae import DecoderTinyTemporal
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video


decoder = DecoderTinyTemporal(
    block_out_channels=(192,192,192,192),
    in_channels=4,
    out_channels=3,
    upsampling_scaling_factor=2,
    act_fn="relu",
    upsample_fn="bilinear",
    num_blocks=(3, 2, 2, 1),
).cuda().half()

decoder.load_state_dict(torch.load("tae_temp.pth",
          map_location=torch.device('cuda')), strict=False)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

def infer_vae(latents, *args, **kwargs):
    ret = decoder(latents.mul(pipe.vae.config.scaling_factor),
                           num_frames=min(latents.shape[0], 8)).mul_(2).sub_(1).cpu()
    return [ret]

pipe.vae.decode = infer_vae

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)