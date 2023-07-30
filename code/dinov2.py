from transformers import AutoImageProcessor, Dinov2Model
import torch
from PIL import Image
import numpy as np


def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

path = "/home/wangye/YeProject_bak/openimage/fast_verify/train/data/00a4e2347de87aea.jpg"
img = load_image(path)
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
model = Dinov2Model.from_pretrained("facebook/dinov2-giant")
inputs = image_processor(img, return_tensors="pt")
print(inputs)


with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape))


