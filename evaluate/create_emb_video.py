import copy
import os
import pickle
import time

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from cdfvd import fvd
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline,
                       StableVideoDiffusionPipeline, UNet2DConditionModel)
from diffusers.models.autoencoders import AutoencoderTiny
from diffusers.models.autoencoders.vae import DecoderTinyTemporal
from PIL import Image
from torchvision import transforms

evaluator = fvd.cdfvd(
    "videomae",
    ckpt_path="/home/alexey.buzovkin/metrics_vid/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
)


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation="bilinear"):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False
    )


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


transform_vmae = transforms.Compose([ToFloatTensorInZeroOne(), Resize((224, 224))])


pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
vae = pipe.vae
vae = vae.to("cuda")

tae = AutoencoderTiny(decoder_block_out_channels=(192, 192, 192, 192)).to("cuda").half()
tae.load_state_dict(
    torch.load("ex_svd1.pth", map_location=torch.device("cuda")), strict=False
)

decoder = (
    DecoderTinyTemporal(
        block_out_channels=(192, 192, 192, 192),
        in_channels=4,
        out_channels=3,
        upsampling_scaling_factor=2,
        act_fn="relu",
        upsample_fn="bilinear",
        num_blocks=(3, 2, 2, 1),
    )
    .cuda()
    .half()
)

decoder.load_state_dict(
    torch.load("tae_temp.pth", map_location=torch.device("cuda")), strict=False
)

train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

path_base = "/home/alexey.buzovkin/metrics_vid/data/UCF101/"
path_ori = os.path.join(path_base, "UCF-101")

dt_orig = []
dt_tae192 = []
dt_taetemp = []

feature_gt = []
feature_orig = []
feature_tae = []
feature_taetemp = []

run_flag = 0

mean_gt = np.zeros(
    [
        1408,
    ],
    dtype=np.float64,
)
mean_orig = np.zeros(
    [
        1408,
    ],
    dtype=np.float64,
)
mean_tae = np.zeros(
    [
        1408,
    ],
    dtype=np.float64,
)
mean_taetemp = np.zeros(
    [
        1408,
    ],
    dtype=np.float64,
)

cov_gt = np.zeros([1408, 1408], dtype=np.float64)
cov_orig = np.zeros([1408, 1408], dtype=np.float64)
cov_tae = np.zeros([1408, 1408], dtype=np.float64)
cov_taetemp = np.zeros([1408, 1408], dtype=np.float64)


for folder in [k for k in os.listdir(path_ori) if k.find(".") == -1]:
    with open("metrics.pkl", "wb") as fb:
        pickle.dump(
            {
                "run_flag": run_flag,
                "mean_gt": mean_gt,
                "mean_orig": mean_orig,
                "mean_tae": mean_tae,
                "mean_taetemp": mean_taetemp,
                "cov_gt": cov_gt,
                "cov_orig": cov_orig,
                "cov_tae": cov_tae,
                "cov_taetemp": cov_taetemp,
                "dt_orig": dt_orig,
                "dt_tae": dt_tae192,
                "dt_taetemp": dt_taetemp,
            },
            fb,
        )
    print(folder, len(os.listdir(os.path.join(path_ori, folder))))
    for avi in tqdm.tqdm(os.listdir(os.path.join(path_ori, folder))):
        vidcap = cv2.VideoCapture(os.path.join(path_ori, folder, avi))

        try:
            success, image = vidcap.read()
            center = [int(image.shape[0] / 2), int(image.shape[1] / 2)]
        except:
            continue

        frame_lst = [image]

        while success:
            success, image = vidcap.read()
            if not success:
                break
            frame_lst.append(image)

        frame_lst = frame_lst[: 8 * (len(frame_lst) // 8)]
        frame_lst = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in frame_lst
        ]
        frame_vmae = [
            torch.tensor(np.transpose(np.array(fr.resize((224, 224))), [2, 0, 1]))[None]
            / 255.0
            for fr in frame_lst
        ]
        with torch.no_grad():
            frame_vmae = [
                evaluator.model.forward_features(
                    torch.cat(frame_vmae[8 * i : 8 * (i + 1)], 0)
                    .transpose(0, 1)[None]
                    .cuda()
                )
                .detach()
                .cpu()
                .numpy()
                for i in range(len(frame_lst) // 8)
            ]
            for feagt in frame_vmae:
                mean_gt += feagt.sum(axis=0).astype(np.float64)
                cov_gt += feagt.astype(np.float64).T @ feagt.astype(np.float64)

        frame_lst = torch.cat([train_transforms(img)[None] for img in frame_lst]).cuda()
        with torch.no_grad():
            emb_lst = vae.encode(frame_lst.to(torch.float16)).latent_dist.sample()

        restored_orig = []
        restored_tae = []
        restored_tae_temp = []
        with torch.no_grad():
            for i in range(emb_lst.shape[0] // 8):
                # original vae ###############################
                t0 = time.time()
                dec_ori = vae.decode(emb_lst[8 * i : 8 * (i + 1)], num_frames=8)
                dt_orig.append(time.time() - t0)
                dec_ori = dec_ori.sample.detach().cpu().numpy()
                restored_orig = copy.deepcopy(
                    [
                        Image.fromarray(
                            (
                                255
                                * np.transpose(np.clip((img + 1) / 2, 0, 1), [1, 2, 0])
                            ).astype(np.uint8)
                        )
                        for img in list(dec_ori)
                    ]
                )
                restored_orig_vmae = [
                    torch.tensor(
                        np.transpose(np.array(fr.resize((224, 224))), [2, 0, 1])
                    )[None]
                    / 255.0
                    for fr in restored_orig
                ]
                with torch.no_grad():
                    restored_orig_vmae = [
                        evaluator.model.forward_features(
                            torch.cat(restored_orig_vmae, 0)
                            .transpose(0, 1)[None]
                            .cuda()
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ]
                    for feagt in restored_orig_vmae:
                        mean_orig += feagt.sum(axis=0).astype(np.float64)
                        cov_orig += feagt.astype(np.float64).T @ feagt.astype(
                            np.float64
                        )

                # tae 192 #############################################
                t0 = time.time()
                dec_tae192 = tae.decode(
                    emb_lst[8 * i : 8 * (i + 1)] * vae.config.scaling_factor
                )
                dt_tae192.append(time.time() - t0)
                dec_tae192 = dec_tae192.sample.mul_(2).sub_(1).detach().cpu().numpy()
                restored_tae = copy.deepcopy(
                    [
                        Image.fromarray(
                            (
                                255
                                * np.transpose(np.clip((img + 1) / 2, 0, 1), [1, 2, 0])
                            ).astype(np.uint8)
                        )
                        for img in list(dec_tae192)
                    ]
                )

                restored_tae_vmae = [
                    torch.tensor(
                        np.transpose(np.array(fr.resize((224, 224))), [2, 0, 1])
                    )[None]
                    / 255.0
                    for fr in restored_tae
                ]

                with torch.no_grad():
                    restored_tae_vmae = [
                        evaluator.model.forward_features(
                            torch.cat(restored_tae_vmae, 0).transpose(0, 1)[None].cuda()
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ]
                    for feagt in restored_tae_vmae:
                        mean_tae += feagt.sum(axis=0).astype(np.float64)
                        cov_tae += feagt.astype(np.float64).T @ feagt.astype(np.float64)

                # tae temporal ########################################
                t0 = time.time()
                dec_taetemp = decoder(
                    emb_lst[8 * i : 8 * (i + 1)] * vae.config.scaling_factor,
                    num_frames=8,
                )
                dt_taetemp.append(time.time() - t0)
                dec_taetemp = dec_taetemp.mul_(2).sub_(1).detach().cpu().numpy()
                restored_taetemp = copy.deepcopy(
                    [
                        Image.fromarray(
                            (
                                255
                                * np.transpose(np.clip((img + 1) / 2, 0, 1), [1, 2, 0])
                            ).astype(np.uint8)
                        )
                        for img in list(dec_taetemp)
                    ]
                )
                restored_taetemp_vmae = [
                    torch.tensor(
                        np.transpose(np.array(fr.resize((224, 224))), [2, 0, 1])
                    )[None]
                    / 255.0
                    for fr in restored_taetemp
                ]

                with torch.no_grad():
                    restored_taetemp_vmae = [
                        evaluator.model.forward_features(
                            torch.cat(restored_taetemp_vmae, 0)
                            .transpose(0, 1)[None]
                            .cuda()
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ]
                    feature_taetemp += restored_taetemp_vmae
                    for feagt in restored_taetemp_vmae:
                        mean_taetemp += feagt.sum(axis=0).astype(np.float64)
                        cov_taetemp += feagt.astype(np.float64).T @ feagt.astype(
                            np.float64
                        )

                run_flag += 1
