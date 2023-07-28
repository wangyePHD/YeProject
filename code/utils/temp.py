import os
import shutil

def get_image_names(folder_path):
    image_names = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg')):
            file = file.split('.')[0]
            image_names.append(file)
        else:
            file = file.split('_')[0]
            image_names.append(file)
    return image_names

# 两个文件夹的路径
folder1_path = "/home/wangye/YeProject/openimage/train_data/data"  # 替换为第一个文件夹的路径
folder2_path = "/home/wangye/YeProject/openimage/train_data/mask"  # 替换为第二个文件夹的路径

# 获取两个文件夹中的图像名称列表
image_names_folder1 = get_image_names(folder1_path)
image_names_folder2 = get_image_names(folder2_path)

# 将图像名称列表转换为集合
image_set_folder1 = set(image_names_folder1)
image_set_folder2 = set(image_names_folder2)

# 找到图像名称的交集
image_intersection = image_set_folder1.intersection(image_set_folder2)

# 删除第一个文件夹中不属于交集部分的图像
for image_name in image_names_folder1:
    if image_name not in image_intersection:
        image_path = os.path.join(folder1_path, image_name+".jpg")
        os.remove(image_path)

print("删除完成！")
