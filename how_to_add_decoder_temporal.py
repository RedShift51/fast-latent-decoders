from diffusers.models.autoencoders.vae import DecoderTinyTemporal

# to know the path to diffusers, just type in python console "diffusers", after "import diffusers"

# miniconda3/envs/taming/lib/python3.8/site-packages/diffusers/models/autoencoders/vae.py
from ..unets.unet_3d_blocks import MidBlockTemporalDecoder

# to the end of the file
class DecoderTinyTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
        upsample_fn: str,
    ):
        super().__init__()
        layers = [
          nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
          get_activation(act_fn),
        ]
        self.num_blocks = num_blocks
        self.layers_per_block = 2
        self.mid_block = MidBlockTemporalDecoder(
          num_layers=self.layers_per_block,
          in_channels=block_out_channels[-1],
          out_channels=block_out_channels[-1],
          attention_head_dim=block_out_channels[-1] // 2,
        )

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]
            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor, mode=upsample_fn))
            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                nn.Conv2d(
                  num_channels,
                  conv_out_channel,
                  kernel_size=3,
                  padding=1,
                  bias=is_final_block,
                )
            )
        self.lay_len = len(layers)
        self.layers = layers
        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int
    ) -> torch.Tensor:
        r"""The forward method of the `DecoderTiny` class."""
        # Clamp.
        x = torch.tanh(x / 3) * 3
        for lnum in range(self.lay_len):
            x = self.layers[lnum](x)
            if lnum == 6:
                upscale_dtype = next(iter(self.mid_block.parameters())).dtype
                batch_size = x.shape[0] // num_frames
                image_only_indicator = torch.zeros(batch_size, num_frames, \
                        dtype=x.dtype, device=x.device)
                x = self.mid_block(
                    x,
                    image_only_indicator=image_only_indicator,
                )
                x = x.to(upscale_dtype)
        # scale image from [0, 1] to [-1, 1] to match diffusers convention
        return x.mul(2).sub(1)


# miniconda3/envs/taming/lib/python3.8/site-packages/diffusers/models/autoencoders/__init__.py
from .vae import DecoderTinyTemporal

# miniconda3/envs/taming/lib/python3.8/site-packages/diffusers/models/__init__.py
_import_structure["autoencoders.vae"] = ["DecoderTinyTemporal"]

# initialising
decoder = DecoderTinyTemporal(
    block_out_channels=(192,192,192,192),
    in_channels=4,
    out_channels=3,
    upsampling_scaling_factor=2,
    act_fn="relu",
    upsample_fn="bilinear",
    num_blocks=(3, 2, 2, 1),
)