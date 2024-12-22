import copy
import os
import sys
import time

import cv2
import lpips
import numpy as np
import pandas as pd
import scipy.stats
import torch
import tqdm
from datasets import load_dataset
from diffusers import (AutoencoderKL, AutoPipelineForText2Image, DDPMScheduler,
                       StableDiffusionXLPipeline, UNet2DConditionModel)
from diffusers.models.autoencoders import AutoencoderTiny
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision import transforms

sys.path.append("/home/alexey.buzovkin/metrics/efficientvit")
from efficientvit.ae_model_zoo import (DCAE_HF, REGISTERED_DCAE_MODEL,
                                       AutoencoderKL)
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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


train_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

vae = pipeline_text2image.vae.float()

cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))

dc_ae = DCAE(cfg).decoder.cuda()
dc_ae.load_state_dict(
    torch.load("effvit_sdxl2.pth", map_location=torch.device("cuda")), strict=False
)

loss_fn_vgg = lpips.LPIPS(net="vgg")


data_list = sorted(
    os.listdir("/home/alexey.buzovkin/data_real/lhq"),
    key=lambda x: int(x[: x.find(".")]),
)[:10000]

ssims = []
pnsrs = []
lps = []
ts = []

for i in tqdm.tqdm(data_list):
    img = Image.open("/home/alexey.buzovkin/data_real/lhq/" + i)
    try:
        code = vae.encode(train_transforms(img)[None].cuda())
    except Exception as e:
        print(e, i)
        continue
    code = code.latent_dist.sample()
    t0 = time.time()
    out = dc_ae(code * 0.13025)[0]

    t1 = time.time()
    ts.append(copy.deepcopy(t1 - t0))

    out = out.detach().cpu().numpy()
    out = np.transpose(out, [1, 2, 0])

    # sdxl vae
    out = np.clip((out + 1) / 2, 0, 1)

    out = (out * 255).astype(np.uint8)

    Image.fromarray(out).save("img_rest_eff/" + i)
    img = img.resize((256, 256))
    """
    lps += list(loss_fn_vgg(
        torch.tensor(2 * (np.transpose(out, [2,0,1]) / 255. - 0.5))[None].float(),
        torch.tensor(2 * (np.transpose(img, [2,0,1]) / 255. - 0.5))[None].float()
    ).detach().cpu().numpy().ravel())
    """
    ssims.append(
        structural_similarity(out, np.array(img), data_range=255, channel_axis=2)
    )
    pnsrs.append(cv2.PSNR(out, np.array(img)))


print("ssim", mean_confidence_interval(ssims))
print("pnsr", mean_confidence_interval(pnsrs))
# print("lpips", mean_confidence_interval(lps))
print("times", mean_confidence_interval(ts))
