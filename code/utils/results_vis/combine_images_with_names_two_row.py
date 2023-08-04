
# coding=utf-8

from PIL import Image
import os 
import numpy as np
from PIL import ImageFont
from PIL import ImageDraw
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T

device = torch.device('cuda')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processer = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)

out = "/home/wangye/YeProject_bak/outputs/combine_mul_images"
base = "/home/wangye/YeProject_bak/outputs/v2"
sup = "object_token0"
# 定义文件夹路径和图像名称列表
name_list_5k = ["DINO+MSE+Reg 5K","DINO+MSE 5K","CLIP+MSE+Reg 5K","DINO+MSE+Reg+Cosine 5K","DINO_Giant 5K"]
name_list_10k = ["DINO+MSE+Reg 10K","DINO+MSE 10K","CLIP+MSE+Reg 10K","DINO+MSE+Reg+Cosine 10K","DINO_Giant 10K"]

folder_paths_5k = ['global_mapping_mse_reg5k', 'global_mapping_mse_no_reg5k', 'global_mapping_clip_5k', 'global_mapping_cosine_5k','global_mapping_dino_giant_5k']
folder_paths_10k = ['global_mapping_mse_reg10k', 'global_mapping_mse_no_reg10k', 'global_mapping_clip_10k', 'global_mapping_cosine_10k','global_mapping_dino_giant_10k']


image_names = os.listdir(os.path.join(base,folder_paths_5k[0],sup))

# 设置每个图像的宽度和高度
image_width = 224
image_height = 224


target = None

simi_5K_list = []
simi_10K_list = []
# 在组合图像中插入每个子图
for img_path in tqdm(image_names):
    # 创建一个新的空白图像，用于组合图像
    combined_image = Image.new('RGB', ((len(folder_paths_5k)+1) * image_width, 2 * image_height + 100)) 
    simi_5k = []
    for i, folder_path in enumerate(folder_paths_5k):
        
        image_path = os.path.join(base,folder_path,sup,img_path)
        img = Image.open(image_path).convert('RGB')
        img_numpy = np.array(img)
        img_numpy_syn = img_numpy[:,0:512,:] 
        target = img_numpy[:,512:-1:]
        target = Image.fromarray(target)
        target = target.resize((image_width, image_height))
        image = Image.fromarray(img_numpy_syn)
        image = image.resize((image_width, image_height))
        image_clip = processer(image,return_tensors="pt")["pixel_values"]
        target_clip = processer(target,return_tensors="pt")["pixel_values"]
        image_clip = image_clip.to(device)
        target_clip = target_clip.to(device)
        with torch.no_grad():
            embedding_img = model.get_image_features(image_clip)
            embedding_target = model.get_image_features(target_clip)
            similarity_score = torch.nn.functional.cosine_similarity(embedding_img, embedding_target)
        similarity_score = similarity_score.detach().cpu().item()
        simi_5k.append(similarity_score)
        
        embedding_img = embedding_img.cpu().numpy()
        combined_image.paste(image, (i * image_width, 0))
        image_name = name_list_5k[i]
        text = image_name+" "+str(similarity_score)[0:5]
        print(text)
        font_size = 200
        draw = ImageDraw.Draw(combined_image)
        draw.text((i * image_width + 50, image_height + 15), text, (255, 255, 255),font_size=font_size)
    
    simi_5K_list.append(simi_5k)
    
    combined_image.paste(target, ((len(folder_paths_5k)) * image_width, 0))
    draw = ImageDraw.Draw(combined_image)
    draw.text(((len(folder_paths_5k)) * image_width + 80, image_height + 15), 'target', (255, 255, 255),font_size=font_size)
    
    simi_10k = []
    for i, folder_path in enumerate(folder_paths_10k):
        image_path = os.path.join(base,folder_path,sup,img_path)
        img = Image.open(image_path).convert('RGB')
        img_numpy = np.array(img)
        img_numpy_syn = img_numpy[:,0:512,:] 
        target = img_numpy[:,512:-1:]
        target = Image.fromarray(target)
        target = target.resize((image_width, image_height))
        image = Image.fromarray(img_numpy_syn)
        image = image.resize((image_width, image_height))
        image_clip = processer(image,return_tensors="pt")["pixel_values"]
        target_clip = processer(target,return_tensors="pt")["pixel_values"]
        image_clip = image_clip.to(device)
        target_clip = target_clip.to(device)
    
        with torch.no_grad():
            embedding_img = model.get_image_features(image_clip)
            embedding_target = model.get_image_features(target_clip)
            similarity_score = torch.nn.functional.cosine_similarity(embedding_img, embedding_target)
        similarity_score = similarity_score.detach().cpu().item()
        simi_10k.append(similarity_score)
        
        combined_image.paste(image, (i * image_width, image_height+40))

        image_name = name_list_10k[i]
        text = image_name+" "+str(similarity_score)[0:5]
        font_size = 200
       
        draw = ImageDraw.Draw(combined_image)
        draw.text((i * image_width + 50, 2*image_height+60), text, (255, 255, 255),font_size=font_size)
    simi_10K_list.append(simi_10k)
    
    combined_image.paste(target, ((len(folder_paths_10k)) * image_width, image_height+40))
    draw = ImageDraw.Draw(combined_image)
    draw.text(((len(folder_paths_10k)) * image_width + 80, 2*image_height+60), 'target', (255, 255, 255),font_size=font_size)
    # 保存组合后的图像
    combined_image.save(os.path.join(out,img_path))
    # combined_image.save('./combined_image.jpg')
    # break
    

simi_5K_array = np.array(simi_5K_list)
simi_10K_array = np.array(simi_10K_list)

for i in range(len(folder_paths_10k)):
    print(simi_5K_array[:,i].mean())

print("-----"*20)

for i in range(len(folder_paths_10k)):
    print(simi_10K_array[:,i].mean())