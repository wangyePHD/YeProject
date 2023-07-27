#!/bin/bash
'''
这个脚本用来将一个文件夹中的大量图像，分别存储到不同的文件夹中，共10个文件夹。每个文件夹中有2000张图像
'''
source_folder="/home/wangye/fiftyone/open-images-v6/test/data"
target="/home/wangye/YeProject/openimage" 
num_target_folders=10
images_per_folder=2000

# Create target folders if they don't exist
for ((i=1; i<=num_target_folders; i++)); do
  target_folder="$target/target_folder${i}"
  mkdir -p "$target_folder"
done

# Move image files to target folders
index=1
folder_index=1
for image_file in "$source_folder"/*; do
  target_folder="$target/target_folder$folder_index"
  cp "$image_file" "$target_folder/"
  index=$((index + 1))

  # Check if we reached the images_per_folder limit
  if [ $((index % images_per_folder)) -eq 0 ]; then
    folder_index=$((folder_index % num_target_folders + 1))
  fi
done
