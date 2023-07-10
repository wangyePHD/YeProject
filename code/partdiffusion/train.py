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
from utils import parse_args

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



def train():
    
    # note parsing the arguments  
    args = parse_args()
    # note define the accelerator
    accelerator = define_accelerator(args)
    # note define logger
    define_logger(args, accelerator)
    
    logger.info(accelerator.state, main_process_only=True)

    

train()


