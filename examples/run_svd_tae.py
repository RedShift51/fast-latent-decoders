from diffusers.models.autoencoders import AutoencoderTiny

import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()


vae = AutoencoderTiny(decoder_block_out_channels=(192, 192, 192, 192)).to("cuda").half()
vae.load_state_dict(torch.load("ex_svd1.pth", map_location=torch.device('cuda')), strict=False)

def infer_vae(latents, *args, **kwargs):
    ret = vae.decode(latents.mul(pipe.vae.config.scaling_factor))[0].mul_(2).sub_(1).cpu()
    return [ret]

pipe.vae.decode = infer_vae


# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)