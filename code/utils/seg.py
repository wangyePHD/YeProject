import sys
sys.path.append("/home/wangye/YeProject/ext/Grounded-Segment-Anything")
import argparse
import os
import copy
import shutil
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# note Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from tqdm import tqdm
# segment anything
from segment_anything import (
    build_sam,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(img_name, raw_img,output_dir, mask, box):
    value = 0  # 0 for background
    
    # note save mask image
    mask_img = torch.zeros(mask.shape[-2:])
    mask_img[mask.cpu().numpy()[0] == True] = 1
    mask_img = Image.fromarray(255*mask_img.cpu().numpy().astype('uint8')).convert('RGB')
    mask_img.save(os.path.join(output_dir, f'{img_name}_bg.png'))
    
    # note cut out for part 
    idx = 0
    
    box = box.numpy().tolist()
    l_x, l_y, r_x, r_y = box
    l_x, l_y, r_x, r_y = int(l_x), int(l_y), int(r_x), int(r_y)
    
    
    mask = torch.unsqueeze(mask,dim=-1)
    mask = torch.concat([mask,mask,mask],dim=-1)
    # cutout = raw_img[l_y:r_y,l_x:r_x,:]

    # plt.imshow(cutout)
    # plt.axis('off')
    # cut_out_dir = os.makedirs(os.path.join(output_dir,"cut_out"))
    # plt.savefig(os.path.join(cut_out_dir, 'cut_out_'+str(image_name)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    # idx+=1
    
    # note write to json file
    txt_line = f"{image_name}${','.join([str(x) for x in box])}\n"
    with open(os.path.join(output_dir, 'mask_box.txt'), 'a+') as f:
        f.write(txt_line)
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--data", type=str, required=True, help="path to img file")
    args = parser.parse_args()

    # note cfg
    config_file = "/home/wangye/Grounded-Segment-Anything-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "/home/wangye/YeProject/ckpts_ext/groundingdino_swint_ogc.pth"
    sam_checkpoint = "/home/wangye/YeProject/ckpts_ext/sam_vit_h_4b8939.pth"
    data_path = args.data
    text_prompt = "car"
    output_dir = "/home/wangye/YeProject/openimage/processed"
    box_threshold = 0.3
    text_threshold = 0.25
    device = "cuda"

    # * load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # * initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    # * make dir
    os.makedirs(output_dir, exist_ok=True)

    for path in tqdm(os.listdir(data_path)):
        
        image_name = path.split('.')[0]
        image_path = os.path.join(data_path, path)
        # * load image
        image_pil, image = load_image(image_path)
        # * run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        
        # note judge the boxes_filt's len is 0, if 0, the image is not car.
        if len(boxes_filt) == 0:
            print(f"{image_name} is not car")
            continue
               
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

       
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
       
        
        # note 选择面积最大的subject，视觉上满足最主要的物体特性
        area_list = []
        for box in boxes_filt:
            area = (box[2] - box[0]) * (box[3] - box[1])
            area_list.append(area.item())
        idx = np.argmax(area_list)
        mask = torch.unsqueeze(masks[idx],dim=0)
        box = boxes_filt[idx]
        
        save_mask_data(image_name, image, output_dir, mask, box)






