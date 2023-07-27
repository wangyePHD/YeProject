import os
import random
from shutil import move
from PIL import Image  # 如果使用PIL库
# import cv2  # 如果使用opencv-python库

source_folder_img = "/home/wangye/YeProject/openimage/train_data/data"  # 源文件夹路径
source_folder_mask = "/home/wangye/YeProject/openimage/train_data/mask"  # 源文件夹路径
destination_folder_img = "/home/wangye/YeProject/openimage/test_data/data"  # 目标文件夹路径
destination_folder_mask = "/home/wangye/YeProject/openimage/test_data/mask"  # 目标文件夹路径
num_images_to_select = 200  # 随机选择的图像数量

# 获取源文件夹下所有图像文件的路径
image_files = [os.path.join(source_folder_img, file) for file in os.listdir(source_folder_img) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# 随机选择num_images_to_select数量的图像文件
selected_image_files = random.sample(image_files, num_images_to_select)

# 将选中的图像文件移动到目标文件夹
for image_file in selected_image_files:
    file_name = os.path.basename(image_file)
    destination_path = os.path.join(destination_folder_img, file_name)
    move(image_file, destination_path)


# 将选中的图像对应的mask移动到目标文件夹
for image_file in selected_image_files:
    file_name = os.path.basename(image_file)
    mask_name = file_name.split('.')[0]+'_bg.png'
    mask_path = os.path.join(source_folder_mask,mask_name)
    destination_path = os.path.join(destination_folder_mask, mask_name)
    move(mask_path, destination_path)