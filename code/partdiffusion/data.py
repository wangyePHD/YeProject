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
        input_ids = self.tokenizer.encode(caption)
        max_len = self.tokenizer.model_max_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        else:
            input_ids += [self.tokenizer.pad_token_id] * (max_len - len(input_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.unsqueeze(0)

        return input_ids
    
    def __getitem__(self, idx):

        json_item = self.data[idx]
        raw_image = json_item['raw_image']
        # ! load part image, note that we just use a part image
        part_image_list = json_item['part_image_list']
        mask_json = json_item['mask_json']
        part_mask_list = json_item['part_mask_list']
        caption = json_item['caption']

        # note load raw image and  part image
        raw_image = Image.open(raw_image).convert('RGB')
        part_image = Image.open(part_image_list[0])

        if self.transform is not None:
            raw_image = self.transform(raw_image)
            part_image = self.transform(part_image)

        input_ids = self._tokenize_caption(caption)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        raw_image = raw_image.to(self.device)
        part_image = part_image.to(self.device)
        input_ids = input_ids.to(self.device)
        

        return raw_image, part_image, input_ids







