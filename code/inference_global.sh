export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='/home/wangye/YeProject_bak/openimage/v1/test_data/mixed'

CUDA_VISIBLE_DEVICES=1 python inference_global_dinov2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir="../outputs/global_mapping_10k"  \
  --suffix="object" \
  --token_index="0" \
  --template="a photo of a S" \
  --global_mapper_path="/home/wangye/YeProject_bak/elite_experiments/global_mapping_1000/mapper_010000.pt" \
  --seed=42

