#!/bin/bash

# Replace "/path/to/destination_folder" with the actual destination folder path.
destination_folder="/home/wangye/fiftyone/open-images-v6/test/data"

# Array containing the 10 source folders.
source_folders=("target_folder1" "target_folder2" "target_folder3" "target_folder4" "target_folder5" "target_folder6" "target_folder7" "target_folder8" "target_folder9" "target_folder10")

# Loop through each source folder and copy image files to the destination folder.
for folder in "${source_folders[@]}"; do
  cp "$folder"/* "$destination_folder/"
done
