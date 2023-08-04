
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR1='/home/wangye/YeProject_bak/openimage/fast_verify/train'
export DATA_DIR2='/home/wangye/YeProject_bak/openimage/fast_verify/val'

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file multi_gpu.json --main_process_port 25666 train_global_dinov2_giant.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR1 \
  --val_data_dir=$DATA_DIR2 \
  --image_encoder="facebook/dinov2-giant" \
  --placeholder_token="S" \
  --resolution_W=512 \
  --resolution_H=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=200000 \
  --learning_rate=1e-06 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="../elite_experiments/global_mapping_1000_divo_giant_200K_hf_map" \
  --save_steps 200
  
