export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='/home/wangye/YeProject/openimage/test_data/mixed'

CUDA_VISIBLE_DEVICES=1 python inference_global_dinov2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir="../outputs/global_mapping"  \
  --suffix="object" \
  --token_index="0" \
  --template="a photo of a S" \
  --global_mapper_path="/home/wangye/YeProject/elite_experiments/global_mapping_20230727_896_384_res/mapper_005000.pt" \
  --seed=42

