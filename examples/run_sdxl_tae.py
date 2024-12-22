from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
import cv2
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms
import tqdm
from diffusers import AutoPipelineForText2Image
import torch
from diffusers.models.autoencoders import AutoencoderTiny
from collections import OrderedDict

vae = AutoencoderTiny(decoder_block_out_channels=(192, 192, 192, 192)).to("cuda").half()
vae.load_state_dict(torch.load("ex_v6.pth", map_location=torch.device('cuda')), strict=False)

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

def infer_vae(latents, *args, **kwargs):
    ret = vae.decode(latents.mul(pipeline_text2image.vae.config.scaling_factor))[0].mul_(2).sub_(1).cpu()
    return [ret]

pipeline_text2image.vae.decode = infer_vae

prompt = "Three smiling girls walking outside"
generator = torch.manual_seed(10)
image = pipeline_text2image(prompt=prompt).images[0]
image.save("3_smiling_girls_walking_outside/vae96_new5.png")