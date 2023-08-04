from typing import Tuple
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel,ControlNetModel
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


# vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
# text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
# unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# # print(unet.up_blocks)
# # del unet.up_blocks

# torch_device = "cuda"
# vae.to(torch_device)
# text_encoder.to(torch_device)
# unet.to(torch_device)

# prompt = ["a photograph of an astronaut riding a horse"]
# height = 512  # default height of Stable Diffusion
# width = 512  # default width of Stable Diffusion
# num_inference_steps = 25  # Number of denoising steps
# guidance_scale = 7.5  # Scale for classifier-free guidance
# generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
# batch_size = len(prompt)

# from diffusers import PNDMScheduler

# scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# text_input = tokenizer(
#     prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
# )

# with torch.no_grad():
#     text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# max_length = text_input.input_ids.shape[-1]
# uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
# uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

# text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


# latents = torch.randn(
#     (batch_size, unet.in_channels, height // 8, width // 8),
#     generator=generator,
# )
# latents = latents.to(torch_device)

# latents = latents * scheduler.init_noise_sigma



# from tqdm.auto import tqdm

# scheduler.set_timesteps(num_inference_steps)

# for t in tqdm(scheduler.timesteps):
#     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
#     latent_model_input = torch.cat([latents] * 2)

#     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#     # predict the noise residual
#     with torch.no_grad():
#         import ipdb
#         ipdb.set_trace()
#         unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
#     break
#     # perform guidance
#     # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#     # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#     # # compute the previous noisy sample x_t -> x_t-1
#     # latents = scheduler.step(noise_pred, t, latents).prev_sample




def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
    
    
