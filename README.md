# Fast Latent Decoders for Diffusion Models

[![License: arXiv](https://img.shields.io/badge/License-arXiv-yellow.svg)](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)

This repository contains **lightweight and fast decoders** for latent diffusion pipelines in both image and video generation. The work is described in the accompanying paper:

> **Toward Lightweight and Fast Decoders for Diffusion Models in Image and Video Generation**  
> [**Alexey Buzovkin**](mailto:alexey.buzovkin@gmail.com), [**Evgeny Shilov**](mailto:jvshilov@gmail.com)

We propose smaller decoder architectures (including **TAE-192** and **EfficientViT** variants) to replace the default VAE decoder in Stable Diffusion or Stable Video Diffusion. Our results demonstrate **speed-ups** (up to 20× in decoding sub-module alone), moderately reduced memory, and acceptable perceptual quality across multiple datasets (COCO, UCF101, LAION, etc.).


## Table of Contents
- [Features](#features)
- [Repository Overview](#repository-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Image Decoding with a Lightweight Decoder](#1-image-decoding-with-a-lightweight-decoder)
  - [2. Video Decoding with a Lightweight Decoder](#2-video-decoding-with-a-lightweight-decoder)
  - [3. Evaluating Decoding Performance](#3-evaluating-decoding-performance)
- [How to Add a Temporal Decoder](#how-to-add-a-temporal-decoder)
- [Examples](#examples)
- [Citation](#citation)
- [License](#license)
- [References](#references)


## Features
- **Lightweight Decoders**  
  - \(\textbf{TAE-192}\): A small autoencoder derived from Taming Transformers + Diffusers.  
  - **EfficientViT-based**: A compressed decoder variant using attention blocks from EfficientViT.
- **Video Support**  
  - Integrates spatiotemporal attention in the decoder (e.g., `DecoderTinyTemporal`) for improved video consistency.
- **Consistent Interfaces**  
  - Compatible with the Hugging Face Diffusers library for both **Stable Diffusion** and **Stable Video Diffusion**.
- **Performance Evaluation**  
  - Scripts to measure runtime, SSIM, PSNR, and compute advanced video metrics (via [VideoMAE V2](https://github.com/MCG-NJU/VideoMAE)).
- **Easy Integration**  
  - Drop-in replacements for the default SDXL or video decoders, requiring minimal code changes in existing pipelines.


## Repository Overview
Below is a high-level view of the repository structure:

```bash
fast-latent-decoders/
├── env_decoders.yml          # Conda environment specification
├── evaluate/
│   ├── measure.py            # Measures reconstruction metrics (SSIM, PSNR, etc.) for TAE decoders
│   ├── measure_eff.py        # Measures reconstruction metrics for EfficientViT-based decoder
│   ├── create_emb_video.py   # Computes video embeddings using VideoMAE for advanced metrics
│   └── calc_dist_video.py    # Calculates FVD/Video FID-like scores from the embedded stats
├── examples/
│   ├── run_sdxl_tae.py       # Example: replacing SDXL decoder with TAE in an image pipeline
│   ├── run_svd_tae.py        # Example: replacing Stable Video Diffusion decoder with TAE
│   ├── run_svd_tae_temp.py   # Example: using the TAE Temporal decoder for video
│   └── run_svd_effvit.py     # Example: using the EfficientViT-based decoder in video
└── how_to_add_decoder_temporal.py
```

- **`env_decoders.yml`**: A YAML file for creating a Conda environment with all required Python dependencies.  
- **`evaluate/`**: Scripts for measuring performance and quality of the reconstructed images/videos:
  - `measure.py`, `measure_eff.py`: Evaluate image-level metrics (SSIM, PSNR) and (optionally) LPIPS.  
  - `create_emb_video.py`, `calc_dist_video.py`: Evaluate advanced video metrics (like content-debiased Fréchet Video Distance) via VideoMAE V2 embeddings.  
- **`examples/`**: Ready-to-run scripts demonstrating how to load or integrate our decoders into:
  - **Stable Diffusion XL** for images, or  
  - **Stable Video Diffusion** for videos.  
- **`how_to_add_decoder_temporal.py`**: Annotated guide for extending the Hugging Face Diffusers library to support a temporal-aware decoder.


## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/RedShift51/fast-latent-decoders.git
   cd fast-latent-decoders
   ```

2. **Create** the Conda environment (recommended):
   ```bash
   conda env create -f env_decoders.yml
   conda activate diff10
   ```

3. **(Optional)** Manually install additional dependencies if needed:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have [PyTorch](https://pytorch.org/) with CUDA or the relevant acceleration backend installed.


## Usage

### 1. Image Decoding with a Lightweight Decoder

For **Stable Diffusion XL**:

1. **Train** or **download** our lightweight decoder checkpoint (e.g., `ex_v6.pth` for TAE-192).
2. **Replace** the default VAE decode method as shown in `examples/run_sdxl_tae.py`:
   ```python
   from diffusers.models.autoencoders import AutoencoderTiny

   # Load TAE-192
   vae_tiny = AutoencoderTiny(decoder_block_out_channels=(192, 192, 192, 192)).to("cuda").half()
   vae_tiny.load_state_dict(torch.load("ex_v6.pth"))

   def infer_vae(latents, *args, **kwargs):
       # scale latents, decode, and map to [-1, 1]
       ret = vae_tiny.decode(latents * scaling_factor)[0].mul_(2).sub_(1)
       return [ret.cpu()]

   # Patch the pipeline
   pipeline.vae.decode = infer_vae
   # Run generation or reconstruction
   ```

3. **Generate or reconstruct** images using the new tiny decoder with minimal overhead.  

### 2. Video Decoding with a Lightweight Decoder

For **Stable Video Diffusion** (e.g., `stabilityai/stable-video-diffusion-img2vid-xt`):

1. **Load** the video pipeline and our small decoder (e.g., `tae_temp.pth`).
2. **Override** the `decode()` method with a function that calls your custom decoder.  
   See `examples/run_svd_tae_temp.py` for a detailed snippet:
   ```python
   from diffusers.models.autoencoders.vae import DecoderTinyTemporal

   decoder = DecoderTinyTemporal(...).cuda().half()
   decoder.load_state_dict(torch.load("tae_temp.pth"))

   def infer_video_vae(latents, *args, **kwargs):
       return [decoder(latents * scaling_factor, num_frames=8).cpu()]

   pipe.vae.decode = infer_video_vae
   # Now generate frames
   ```

### 3. Evaluating Decoding Performance

- **Image metrics** (`SSIM`, `PSNR`, etc.):
  ```bash
  cd evaluate
  python measure.py       # for TAE decoders
  python measure_eff.py   # for EfficientViT-based decoders
  ```
  These scripts read images from a specified folder, encode/decode them, and compute the metrics.

- **Video metrics** (using `VideoMAE V2` embeddings):
  ```bash
  python create_emb_video.py   # extracts embeddings for generated videos
  python calc_dist_video.py    # computes FVD-like metrics
  ```
  Ensure you have the correct paths to your generated videos (or frames) and the reference videos.


## How to Add a Temporal Decoder
See [`how_to_add_decoder_temporal.py`](how_to_add_decoder_temporal.py) for a step-by-step tutorial on modifying **Diffusers** to integrate a temporal-aware block, such as `DecoderTinyTemporal`. This guide walks through the code changes needed to:
- Insert spatiotemporal attention blocks,
- Extend the 2D decoder to handle 3D conv or attention across frames,
- Register the new class within the Hugging Face library structure.

This method is particularly useful if you want to customize your video decoder beyond the existing examples.


## Examples
- **`examples/run_sdxl_tae.py`**  
  Replace the SDXL VAE with the TAE-192 decoder for faster image generation.  

- **`examples/run_svd_tae.py`**  
  Demonstrates integrating a TAE-192 decoder into a Stable Video Diffusion pipeline (frame-wise).  

- **`examples/run_svd_tae_temp.py`**  
  Uses the **temporal** TAE (`DecoderTinyTemporal`) for consistent decoding across frames.  

- **`examples/run_svd_effvit.py`**  
  Shows how to load and run the **EfficientViT-based** decoder in a video diffusion pipeline.  

Each script documents how to load the checkpoint, patch the pipeline, and generate outputs.


## Citation
If you find this project useful in your research, please cite our paper:

```
@article{buzovkin2024fastdecoders,
  title     = {Toward Lightweight and Fast Decoders for Diffusion Models in Image and Video Generation},
  author    = {Buzovkin, Alexey and Shilov, Evgeny},
  journal   = {arXiv preprint arXiv:xxxx.xxxx},
  year      = {2024},
  url       = {https://github.com/RedShift51/fast-latent-decoders}
}
```

You can also link back to this GitHub repository to help others easily find and use these decoders.


## License
This work is licensed under the arXiv.org perpetual, non-exclusive license.

By submitting this work, the author(s) grant arXiv.org a perpetual, non-exclusive license to distribute this work. The author(s) retain all other rights, including copyright, to this work.


## References
- **Stable Diffusion XL**:  
  [https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)  
- **Stable Video Diffusion**:  
  [https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  
- **Taming Transformers**:  
  [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)  
- **EfficientViT**:  
  [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)  
- **VideoMAE V2**:  
  [https://github.com/MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE)


---

For additional details, usage examples, or potential issues, please open a GitHub Issue or contact us via email. We welcome contributions, suggestions, and collaborations to further improve fast and efficient decoding strategies in diffusion-based generative models!