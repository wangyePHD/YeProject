import os

def get_image_names(folder_path):
    image_names = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_names.append(file)
    return image_names

def save_to_txt(image_names, output_file):
    with open(output_file, 'w') as f:
        for name in image_names:
            f.write(name + '\n')

# 指定文件夹路径
folder_path = "/home/wangye/YeProject/openimage/train_data/data"  # 将其替换为你的文件夹路径

# 获取文件夹下所有图像的名称
image_names = get_image_names(folder_path)

# 将图像名称保存到txt文件
output_file = "/home/wangye/YeProject/openimage/train_data/mask_box.txt"
save_to_txt(image_names, output_file)
