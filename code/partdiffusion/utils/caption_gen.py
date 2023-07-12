
'''
run this file:
conda activate fastcomposer
python caption_gen.py
'''



from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
path = "/home/wangye/YeProject/data/demo_dataset/test"
folder_list = os.listdir(path)
model.to(device)
for folder in tqdm(folder_list):
    raw_image = os.path.join(os.path.join(path,folder), folder+".jpg")
    print(raw_image)
    raw_image = Image.open(raw_image).convert('RGB')
    inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[3].strip()
    print(generated_text)
    
    

