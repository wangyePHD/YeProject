import os
import json 
from tqdm import tqdm

part_num = 1
path = "/home/wangye/YeProject/data/demo_dataset/train"
folder_list = os.listdir(path)
data_all = []
for folder in tqdm(folder_list):
    raw_image = os.path.join(os.path.join(path,folder), folder+".jpg")
    part_image_list = []
    for i in range(part_num):
        part_img = os.path.join(os.path.join(path,folder),"cut_out_"+str(i)+".jpg")
        part_image_list.append(part_img)
    json_file = os.path.join(os.path.join(path,folder),"mask.json")
    
    part_mask_list = []
    for i in range(part_num):
        part_mask = os.path.join(os.path.join(path,folder),"wheel._mask_"+str(i)+".jpg")
        part_mask_list.append(part_mask)

    caption_path = os.path.join(os.path.join(path,folder),"caption.txt")
    file = open(caption_path,'r')
    caption = file.read()

    dict_data = {
        'raw_image': raw_image,
        'part_image_list': part_image_list,
        'mask_json': json_file,
        'part_mask_list': part_mask_list,
        'caption': caption
    }

    data_all.append(dict_data)

# note data_all 写入json文件 
output_path = "/home/wangye/YeProject/data/demo_dataset/train_data.json"

# 打开文件并将 data_all 写入其中
with open(output_path, 'w') as json_file:
    json.dump(data_all, json_file)

# 打印成功消息
print("数据已成功写入 JSON 文件:", output_path)

   
   

        
