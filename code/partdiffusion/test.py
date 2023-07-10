import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from transformers.models.clip.modeling_clip import (
    _expand_mask,
    CLIPTextTransformer,
    CLIPPreTrainedModel,
    CLIPModel,
)

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path,subfolder="text_encoder",revision=None)
print(model)