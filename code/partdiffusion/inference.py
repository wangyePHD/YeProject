import os
import torch
import numpy as np
import types
import itertools


from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from accelerate.utils import set_seed
from accelerate import Accelerator
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from data import DemoDataset
from model import PartDiffusion
from utils.utils import parse_args
from train import define_transforms
from our_inference import PartDiffusionInference
from torchvision import transforms

@torch.no_grad()
def main():
    args  = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": # note 此处执行
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": # note 此处不执行
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    model = PartDiffusion.from_pretrained(args)
    ckpt_name = "pytorch_model.bin"
    model.load_state_dict(
        torch.load(Path(args.finetuned_model_path) / ckpt_name, map_location="cpu")
    )
    model = model.to(device=accelerator.device, dtype=weight_dtype)

    pipe.unet = model.unet

    if args.enable_xformers_memory_efficient_attention: # note 不执行
        pipe.unet.enable_xformers_memory_efficient_attention()
    
    pipe.text_encoder = model.text_encoder
    pipe.image_encoder = model.image_encoder

    pipe.inference = types.MethodType(
        PartDiffusionInference, pipe
    )
    
    del model

    pipe = pipe.to(accelerator.device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    file = "/home/wangye/YeProject/data/demo_dataset/test_data.json"
    train_transforms = define_transforms(flag='Train')
    train_dataset = DemoDataset(file,tokenizer,accelerator.device,train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,1,shuffle=False)

    path = "/home/wangye/YeProject/code/outputs"
    for i, batch in enumerate(tqdm(train_dataloader)):
        
        raw_image, part_image, input_ids = batch
        raw_image = raw_image.to(accelerator.device)
        part_image = part_image.to(accelerator.device)
        input_ids = input_ids.to(accelerator.device)
        part_image = part_image.to(pipe.image_encoder.parameters().__next__().dtype)
        
        input_ids = torch.squeeze(input_ids,dim=1)
        text_embeddings = pipe.text_encoder(input_ids)[0]
        part_embeddings = pipe.image_encoder(part_image)
        part_embeddings = torch.unsqueeze(part_embeddings,1)
        
        # multi_modal_fusion_embedding = torch.concat([text_embeddings,part_embeddings],dim=1)
       
        image = pipe.inference(
            prompt_embeds = text_embeddings,
            part_embeddings = part_embeddings
        ).images
        
        # image = toPIL(image[0])
        image[0].save(path+'/'+str(i)+'.png')
        
main()