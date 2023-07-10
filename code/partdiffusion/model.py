import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel
from transformers.configuration_utils import PretrainedConfig
from typing import Any, Optional, Tuple, Union, Dict, List
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    _expand_mask,
    CLIPTextTransformer,
    CLIPPreTrainedModel,
    CLIPModel,
)

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"


class PartDiffusionPostFusionModule(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    def forward():
        pass

class PartDiffusionTextEncoder(CLIPPreTrainedModel):
    _build_causal_attention_mask = CLIPTextTransformer._build_causal_attention_mask
    @staticmethod
    def from_pretrained(model_name_or_path, subfolder,**kwargs):
        model = CLIPTextModel.from_pretrained(model_name_or_path,subfolder=subfolder,**kwargs)
        text_model = model.text_model
        
        return  PartDiffusionTextEncoder(text_model)
    
    def __init__(self, text_model):
        super().__init__(text_model.config)
        self.config = text_model.config
        # note: 1 token_embedding + position_embedding
        self.embeddings = text_model.embeddings
        # note: 2 encoder
        self.encoder = text_model.encoder
        # note: 3 layernorm
        self.final_layer_norm = text_model.final_layer_norm
    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            raise ValueError("The input_ids should not be None!")
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # note text token id --> text embedding
        hidden_states = self.embeddings(input_ids)

        bsz, seq_len = input_shape
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        ).to(hidden_states.device)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        # note text encoder extracting feature
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                dim=-1
            ),
        ]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    

class PartDiffusionImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path
    ):
        # note global_model_name_or_path:  openai/clip-vit-large-patch14
        model = CLIPModel.from_pretrained(global_model_name_or_path)
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return PartDiffusionImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )
    
    def __init__(self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    # ! TODO
    def forward(self, part_pixel_values):
        b, num_objects, c, h, w = part_pixel_values.shape
        part_pixel_values = part_pixel_values.view(b * num_objects, c, h, w)
        # note 插值处理 补到224
        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode="bilinear", antialias=True
            )

        part_pixel_values = self.vision_processor(part_pixel_values)
        part_embeds = self.vision_model(part_pixel_values)[1]
        part_embeds = self.visual_projection(part_embeds)
        
        return part_embeds




class PartDiffusion(nn.Module):

    def __init__(self, text_encoder, image_encoder, vae, unet, args):
        super().__init__()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.use_ema = False
        self.ema_param = None
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.revision = args.revision
        self.non_ema_revision = args.non_ema_revision
        self.object_localization = args.object_localization
        self.object_localization_weight = args.object_localization_weight
        self.localization_layers = args.localization_layers
        self.mask_loss = args.mask_loss
        self.mask_loss_prob = args.mask_loss_prob

    @staticmethod
    def from_pretrained(args):
        # note load components

        text_encoder = PartDiffusionTextEncoder.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )

        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder='vae',
            revision=args.revision,
        )

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision,
        )

        image_encoder = PartDiffusionImageEncoder.from_pretrained(
            args.image_encoder_name_or_path
        )

        return PartDiffusion(text_encoder,image_encoder,vae,unet,args)
    

    def forward(self, batch, noise_scheduler):
        
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]
        wheel_pixel = batch['wheel_pixel'] # [3, 224, 224]
 
        # note vae latents 
        vae_dtype = self.vae.parameters().__next__().dtype
        vae_input = pixel_values.to(vae_dtype)

        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # * [batch ,4, 64, 64]
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        )
        
        timesteps = timesteps.long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        # note (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        # note extracting the part features
        wheel_embeds = self.image_encoder(wheel_pixel)
        # note CLIP text encoder for text embedding
        text_embeddings = self.text_encoder(input_ids)[0]
        
        # ! process fusion
        wheel_embeds = torch.unsqueeze(wheel_embeds,dim=1)
        multimodel_fusion_embeds = torch.concat([text_embeddings,wheel_embeds],dim=1)
        # note Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
        
        pred = self.unet(noisy_latents, timesteps, multimodel_fusion_embeds).sample

        # note 只有在满足条件的情况下（通过随机数比较），才会执行这些操作。这样可以在一定概率下引入对象遮罩损失，以提高模型的性能和鲁棒性。cite from chatgpt
        if self.mask_loss and torch.rand(1) < self.mask_loss_prob:
            object_segmaps = batch["object_segmaps"]
            mask = (object_segmaps.sum(dim=1) > 0).float()
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(pred.shape[-2], pred.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred = pred * mask
            target = target * mask

        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        
   
        return_dict = {"denoise_loss": denoise_loss}


# # note test case
# from utils import parse_args
# args = parse_args()
# noise_scheduler = DDPMScheduler.from_pretrained(
#         args.pretrained_model_name_or_path, subfolder="scheduler"
#     )
# model = PartDiffusion.from_pretrained(args)
# input_ids = torch.randint(1,100,(1,77))
# pixel_values = torch.rand((1,3,224,224))
# wheel_pixel = torch.rand((1,1,3,224,224))
# batch = {
#     'input_ids': input_ids,
#     'pixel_values': pixel_values,
#     'wheel_pixel': wheel_pixel
# }
# model(batch,noise_scheduler)

