export MODEL_NAME="CompVis/stable-diffusion-v1-4"
<<<<<<< HEAD
export DATA_DIR='/home/wangye/YeProject/openimage-v6/car'
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file multi_gpu.json --main_process_port 25656 train_global_dinov2.py \

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 25656 train_global_dinov2.py \
=======
export DATA_DIR='../car'
# CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 25656 train_global_dinov2.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file multi_gpu.json --main_process_port 25656 train_global_dinov2.py \
>>>>>>> d24525c448bafb843522b9f5b475a274d7ce8801
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="S" \
  --resolution=512 \
<<<<<<< HEAD
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
=======
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=20000 \
>>>>>>> d24525c448bafb843522b9f5b475a274d7ce8801
  --learning_rate=1e-06 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="../elite_experiments/global_mapping_20230726" \
  --save_steps 200
  