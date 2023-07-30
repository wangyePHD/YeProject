export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='/home/wangye/YeProject_bak/openimage/fast_verify/val'

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file multi_gpu.json --main_process_port 25600 train_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="S" \
  --resolution_W=512 \
  --resolution_H=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=1e-5 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --global_mapper_path "/home/wangye/YeProject_bak/elite_experiments/global_mapping_1000/mapper_010000.pt" \
  --output_dir="./elite_experiments/local_mapping_dubug" \
  --save_steps 200
