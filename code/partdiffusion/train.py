'''
# note accelerator OK
# note 日志代码
# note 初始化模型代码
# note 模型参数设置代码
# note EMA模型代码
# note 优化器设置
# note 数据集
# note 
'''

import logging
import math
import os
import shutil
import torch
import torch.utils.checkpoint
import sys
import datasets
import diffusers
import transformers
import time

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from pathlib import Path
from utils.utils import parse_args

from model import PartDiffusion

logger = get_logger(__name__)


def define_accelerator(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, # note 1
        mixed_precision=args.mixed_precision, # note None
        log_with=args.report_to,
        project_dir=args.logging_dir
    )

    # note Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    return accelerator



def define_logger(args, accelerator):

    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.logging_dir, f"{str_m_d_y_h_m_s}.log")
            ),
        ]
        if accelerator.is_main_process
        else [],
    )

def set_random_seed(args):
    set_seed(args.seed)



def train():
    
    # note parsing the arguments  
    args = parse_args()
    # note define the accelerator
    accelerator = define_accelerator(args)
    # note define logger
    define_logger(args, accelerator)
    
    logger.info(accelerator.state, main_process_only=True)

    # note set random seed
    set_random_seed(args)
    logger.info("Set random seed as "+str(args.seed))

    # note loading model components
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    model = PartDiffusion.from_pretrained(args)
    logger.info("Success for loading components and model!")

    # note judge whether to use mixed precision
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    logger.info("The mixed precision is "+str(weight_dtype))

    # note freeze all params of model
    for param in model.parameters():
        param.requires_grad = False
        param.data = param.data.to(weight_dtype)
    
    logger.info("The requires_grad of all the params is False")
    logger.info("The data type of all the params is "+str(weight_dtype))

    # note unfreeze unet
    model.unet.requires_grad_(True)
    model.unet.to(torch.float32)
    logger.info("Unfreezing the UNet params")

    if args.train_text_encoder: # note 不执行
        model.text_encoder.requires_grad_(True)
        model.text_encoder.to(torch.float32)
        logger.info("Unfreezing the text encoder params")


    if args.train_image_encoder: # note 执行
        if args.image_encoder_trainable_layers > 0:
            for idx in range(args.image_encoder_trainable_layers):
                model.image_encoder.vision_model.encoder.layers[
                    -1 - idx
                ].requires_grad_(True)
                model.image_encoder.vision_model.encoder.layers[-1 - idx].to(
                    torch.float32
                )
            logger.info("Unfreezing the partial params of image encoder")
        else: # note 不执行
            model.image_encoder.requires_grad_(True)
            model.image_encoder.to(torch.float32)
            logger.info("Unfreezing all params of image encoder")

    
    # note Create EMA for the unet.
    if args.use_ema: # note 不执行
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )
        model.load_ema(ema_unet)
        if args.load_model is not None:
            model.ema_param.load_state_dict(
                torch.load(
                    Path(args.load_model) / "custom_checkpoint_0.pkl",
                    map_location="cpu",
                )
            )
    logger.info("Creating EMA for the unet")

    if args.enable_xformers_memory_efficient_attention: # note 不执行
        if is_xformers_available():
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
        logger.info("Enable xformers memory efficient attention")
    
    if args.gradient_checkpointing: # note 不执行
        if args.train_text_encoder:
            model.text_encoder.gradient_checkpointing_enable()
            logger.info("Enable gradient checkpoint for text encoder")

    # * Enable TF32 for faster training on Ampere GPUs,
    # * cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32: # note 执行
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("set torch.backends.cuda.matmul.allow_tf32 True")
    
    if args.scale_lr: # note 不执行
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
        logger.info("scaling the learning rate")


    optimizer_cls = torch.optim.AdamW
    unet_params = list([p for p in model.unet.parameters() if p.requires_grad])
    other_params = list(
        [p for n, p in model.named_parameters() if p.requires_grad and "unet" not in n]
    )
    parameters = unet_params + other_params
    optimizer = optimizer_cls(
        [
            {"params": unet_params, "lr": args.learning_rate * args.unet_lr_scale},
            {"params": other_params, "lr": args.learning_rate},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info("Setup the optimizer")
    
    # ! Define Dataset and dataloader



train()


