export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='/home/wangye/2023_fall_research/ELITE/test_datasets/my/'
CUDA_VISIBLE_DEVICES=0 python inference_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir="../outputs/local_mapping"  \
  --suffix="object" \
  --template="a photo of a S" \
  --llambda="0.8" \
  --global_mapper_path="../ckpts/checkpoints/global_mapper.pt" \
  --local_mapper_path="../ckpts/checkpoints/local_mapper.pt" \
  --seed=42

