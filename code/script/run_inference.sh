CAPTION="a man <|image|> and a man <|image|> are writing codes on the front of computers"
DEMO_NAME="newton_einstein"

CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --mixed_precision=fp16 \
    partdiffusion/inference.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --finetuned_model_path /home/wangye/YeProject/ckpts/models/stable-diffusion-v1-5/Stanford-Car/PartDiffusion/checkpoint-7000 \
    --test_reference_folder data/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir outputs/${DEMO_NAME} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 2 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 5 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --no_object_augmentation
