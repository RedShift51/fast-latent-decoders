import cv2
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms
import copy

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

import sys
sys.path.append("/home/alexey.buzovkin/efficientvit")

from efficientvit.ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL, AutoencoderKL
from efficientvit.models.efficientvit.dc_ae import DCAE, DCAEConfig


cfg_str = (
    "latent_channels=4 "
    "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
    "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[0,4,8,2,2,2] "
    "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
    "decoder.width_list=[128,128,128,256,256,256] decoder.depth_list=[0,5,10,0,0,0] "
    "decoder.norm=[bn2d,bn2d,bn2d,bn2d,bn2d,bn2d] decoder.act=[relu,relu,relu,silu,silu,silu]"
)

from omegaconf import MISSING, OmegaConf

cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))

dc_ae = DCAE(cfg).decoder.cuda()
dc_ae.load_state_dict(torch.load("ex_svd_effvit.pth", map_location=torch.device('cuda')), strict=False)
print(list(dc_ae.state_dict().keys()))

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

def infer_vae(latents, *args, **kwargs):
    ret = dc_ae(latents.mul(pipe.vae.config.scaling_factor).float()).cpu()
    return [ret]

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)