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
from data import DemoDataset

logger = get_logger(__name__)


def define_accelerator(args):
    """
    Define an accelerator object based on the given arguments.

    Parameters:
        args (object): An object containing the arguments for defining the accelerator.

    Returns:
        Accelerator: The accelerator object defined based on the given arguments.
    """

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
    """
    Defines a logger for the program.

    Args:
        args (dict): The arguments for the program.
        accelerator (Accelerator): The accelerator used for the program.

    Returns:
        None
    """
    
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

def define_transforms(flag):

    if flag == 'Train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪为固定大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 随机颜色变换
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪为固定大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 随机颜色变换
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    return transform
    


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
    

    train_transforms = define_transforms(flag="Train")
    test_transforms = define_transforms(flag="Test")
    logger.info("Defining the transforms for test and train data")

    device = accelerator.device
    file = "/home/wangye/YeProject/data/demo_dataset/train_data.json"
    train_dataset = DemoDataset(file,tokenizer,device,train_transforms)
    logger.info("The Dataset length is "+str(train_dataset.__len__()))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,args.train_batch_size)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:# note 不执行
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # note define scheduler for optimizer
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    logger.info("define scheduler for optimizer")
    
    # note Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("Prepare everything with diffuser accelerator")

    if args.use_ema: # note 不执行
        accelerator.register_for_checkpointing(model.module.ema_param)
        model.module.ema_param.to(accelerator.device)

    # note We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps: # note 不执行
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # note Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process: # note 执行
        accelerator.init_trackers("PartDiffusion", config=vars(args))

    
    # note Train!
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} with shape {param.shape}")

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # note Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint: # note 执行
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else: # note 执行
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None: # note 执行
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else: # note 不执行
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
            # move all the state to the correct device
            model.to(accelerator.device)
            if args.use_ema:
                model.module.ema_param.to(accelerator.device)


    # note Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    # note Training
    logger.info("Starting Training !!!")
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        denoise_loss = 0.0
        localization_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            progress_bar.set_description("Global step: {}".format(global_step))

            with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(
                enable_flash = not args.disable_flashattention
            ):
                
                return_dict = model(batch, noise_scheduler)
                loss = return_dict["denoise_loss"]

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_denoise_loss = accelerator.gather(return_dict["denoise_loss"].repeat(args.train_batch_size)).mean()

                denoise_loss += (avg_denoise_loss.item() / args.gradient_accumulation_steps)

                if "localization_loss" in return_dict: # note 执行
                    avg_localization_loss = accelerator.gather(
                        return_dict["localization_loss"].repeat(args.train_batch_size)
                    ).mean()
                    localization_loss += (
                        avg_localization_loss.item() / args.gradient_accumulation_steps
                    )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients: # note 执行
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients: # note 执行
                if args.use_ema: # note 不执行
                    model.module.ema_param.step(model.module.unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "denoise_loss": denoise_loss,
                        "localization_loss": localization_loss,
                    },
                    step=global_step,
                )
                train_loss = 0.0
                denoise_loss = 0.0
                localization_loss = 0.0

                if (
                    global_step % args.checkpointing_steps == 0
                    and accelerator.is_local_main_process
                ):
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    if args.keep_only_last_checkpoint:
                        # Remove all other checkpoints
                        for file in os.listdir(args.output_dir):
                            if file.startswith(
                                "checkpoint"
                            ) and file != os.path.basename(save_path):
                                ckpt_num = int(file.split("-")[1])
                                if (
                                    args.keep_interval is None
                                    or ckpt_num % args.keep_interval != 0
                                ):
                                    logger.info(f"Removing {file}")
                                    shutil.rmtree(os.path.join(args.output_dir, file))

            logs = {
                "l_noise": return_dict["denoise_loss"].detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            if "localization_loss" in return_dict:
                logs["l_loc"] = return_dict["localization_loss"].detach().item()

            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            model.ema_param.copy_to(model.unet.parameters())

        pipeline = model.to_pipeline()
        pipeline.save_pretrained(args.output_dir)
        logger.info("Saving pipeline to " + args.output_dir)
    accelerator.end_training()
    logger.info("Training finished!")



if __name__ == "__main__":

    train()
    



