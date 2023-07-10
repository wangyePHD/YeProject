import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob
import json
from tqdm import tqdm
import numpy as np
import random

from copy import deepcopy
from torch.utils.data import Dataset

from PIL import Image 



class DemoDataset(Dataset):

    def __init__(self,file,tokenizer,device,transform) -> None:
        super().__init__()
        self.file = file
        self.tokenizer = tokenizer
        self.device = device
        self.transform = transform

        with open(file, 'r') as json_file:
            self.data = json.load(json_file)

    def __len__(self):
        return len(self.data)
    

    def _tokenize_caption(self, caption):
        token_id = self.tokenizer(caption)
        return token_id
    
    def __getitem__(self, idx):
        json_item = self.data[idx]
        raw_image = json_item['raw_image']
        part_image_list = json_item['part_image_list']
        mask_json = json_item['mask_json']
        part_mask_list = json_item['part_mask_list']
        caption = json_item['caption']

        # note load raw image
        raw_image = Image.open(raw_image).convert('RGB')
        if self.transform is not None:
            raw_image = self.transform(raw_image)


        # TODO part related image 

        token_ids = self._tokenize_caption(caption)

        raw_image = raw_image.to(self.device)
        token_ids = token_ids.to(self.device)

        return raw_image, token_ids
    








