import copy
import os
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

# from cocodataset import COCODataset


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
vae1 = (
    AutoencoderTiny(decoder_block_out_channels=(192, 192, 192, 192)).to("cuda").half()
)
vae1.load_state_dict(
    torch.load("ex_v6.pth", map_location=torch.device("cuda")), strict=False
)


loss_fn_vgg = lpips.LPIPS(net="vgg")

ssims = []
pnsrs = []
lps = []
ts = []

data_list = sorted(
    os.listdir("/home/alexey.buzovkin/data_real/lhq"),
    key=lambda x: int(x[: x.find(".")]),
)[:10000]

for i in tqdm.tqdm(data_list):
    img = Image.open("/home/alexey.buzovkin/data_real/lhq/" + i)
    try:
        code = vae.encode(train_transforms(img)[None].cuda())
    except:
        print(i)
        continue
    code = code.latent_dist.sample()
    t0 = time.time()
    out = vae1.decode(code.half() * 0.13025)
    t1 = time.time()
    ts.append(copy.deepcopy(t1 - t0))

    out = out.sample[0].detach().cpu().numpy()
    out = np.transpose(out, [1, 2, 0])

    # sdxl vae
    out = np.clip((out + 1) / 2, 0, 1)
    out = (out * 255).astype(np.uint8)

    # img = cv2.resize(np.array(img), \
    #        (256, 256), \
    #        interpolation=cv2.INTER_LINEAR)
    img = img.resize((256, 256))

    # img.save("img_gt/" + i)
    Image.fromarray(out).save("img_rest_tae192/" + i)
    # 1/0
    # t0 = time.time()
    """
    lps += list(loss_fn_vgg(
        torch.tensor(2 * (np.transpose(\
                out, [2,0,1]) / 255. - 0.5))[None].float(),
        torch.tensor(2 * (np.transpose(img, [2,0,1]) / 255. - 0.5))[None].float()
    ).detach().cpu().numpy().ravel())
    """
    # print(time.time()-t0)
    ssims.append(
        structural_similarity(out, np.array(img), data_range=255, channel_axis=2)
    )
    pnsrs.append(cv2.PSNR(out, np.array(img)))


print("ssim", mean_confidence_interval(ssims))
print("pnsr", mean_confidence_interval(pnsrs))
# print("lpips", mean_confidence_interval(lps))
print("times", mean_confidence_interval(ts))
