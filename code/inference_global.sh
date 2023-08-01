export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='/home/wangye/YeProject_bak/openimage/v1/test_data/mixed'
export dino_type='facebook/dinov2-giant'

CUDA_VISIBLE_DEVICES=2 python inference_global_dinov2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir="/home/wangye/YeProject_bak/outputs/v2/global_mapping_dino_giant_20k/"  \
  --suffix="object" \
  --token_index="0" \
  --template="a photo of a S" \
  --mapper_input=1536 \
  --dino_type=$dino_type \
  --global_mapper_path="/home/wangye/YeProject_bak/elite_experiments/global_mapping_1000_divo_giant_200K/mapper_020000.pt" \
  --seed=42

