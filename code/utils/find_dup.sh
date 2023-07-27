#!/bin/bash

# Function to calculate the MD5 hash of a file
calculate_md5() {
  md5sum "$1" | awk '{print $1}'
}

# Function to recursively find duplicate files in a directory
find_duplicate_files() {
  local directory="$1"
  declare -A md5_map

  # Loop through each file in the directory and its subdirectories
  while IFS= read -r -d '' file; do
    # Calculate the MD5 hash of the file
    md5=$(calculate_md5 "$file")

    # Check if the MD5 hash is already in the map (duplicate file)
    if [[ -n "${md5_map[$md5]}" ]]; then
      echo "Duplicate: $file"
      echo "Original: ${md5_map[$md5]}"
      echo
    else
      # Add the MD5 hash and file path to the map
      md5_map[$md5]="$file"
    fi
  done < <(find "$directory" -type f -print0)
}

# Replace "/path/to/folder" with the actual path of the folder you want to search for duplicates.
find_duplicate_files "/home/wangye/YeProject/code/utils"
