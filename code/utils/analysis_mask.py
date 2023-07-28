import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


mask_folder = "/home/wangye/YeProject/openimage/processed"
file_list = []
data_list = []
for mask_img in tqdm(os.listdir(mask_folder)):
    
    if 'png' in mask_img:
        mask_path = os.path.join(mask_folder, mask_img)
        mask = np.array(Image.open(mask_path))    
        mask = np.where(mask > 0, 1, 0)
        data_list.append(np.sum(mask))
        file_list.append(mask_img)
    else:
        pass

# 将列表转换为NumPy数组
data_array = np.array(data_list)


# 绘制箱线图
plt.boxplot(data_array)
plt.savefig("boxplot.png")

# 获取箱线图中的上界（Q3 + 1.5 * IQR）和下界（Q1 - 1.5 * IQR）
q1 = np.percentile(data_array, 25)
q3 = np.percentile(data_array, 75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr


# open a txt file
txt_file = "/home/wangye/YeProject/openimage/processed/mask_img_filter.txt"
os.makedirs("/home/wangye/YeProject/openimage/processed/bad_case", exist_ok=True)
os.makedirs("/home/wangye/YeProject/openimage/processed/good_case", exist_ok=True)
f = open(txt_file,'a+')

# 遍历mask，过滤掉不在上界和下界的mask及对应的图像
idx = 0
for i in tqdm(range(len(file_list))):
    if data_array[i] > upper_bound or data_array[i] < lower_bound:
        print(f"{file_list[i]} 异常！")
        # copy to bad_case folder using shutil
        shutil.copy(os.path.join(mask_folder, file_list[i]), "/home/wangye/YeProject/openimage/processed/bad_case/")
        idx+=1
    else:
        # write the filename into txt file
        f.write(f"{file_list[i]}"+'\n')
        # copy to good case folder using shutil
        shutil.copy(os.path.join(mask_folder, file_list[i]), "/home/wangye/YeProject/openimage/processed/good_case/")
        
print(f"共{idx}个异常图像,{len(data_list)-idx}个正常图像")
