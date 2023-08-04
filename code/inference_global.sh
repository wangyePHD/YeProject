export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='/home/wangye/YeProject_bak/openimage/v1/test_data/mixed'
export dino_type='facebook/dinov2-giant'

CUDA_VISIBLE_DEVICES=2 python inference_global_dinov2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir="/home/wangye/YeProject_bak/outputs/v3/hf_map_condition"  \
  --suffix="object" \
  --token_index="0" \
  --template="a photo of a S" \
  --mapper_input=1536 \
  --dino_type=$dino_type \
  --global_mapper_path="/home/wangye/YeProject_bak/elite_experiments/global_mapping_1000_divo_giant_200K_hf_map/mapper_004000.pt" \
  --pretrained_controlnet_model_path="/home/wangye/YeProject_bak/elite_experiments/global_mapping_1000_divo_giant_200K_hf_map/controlnet_004000.pt" \
  --seed=42

