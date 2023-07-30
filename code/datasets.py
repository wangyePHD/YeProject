from packaging import version
from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
import random
import albumentations as A
import copy
import cv2
import pandas as pd


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()

class CustomDatasetWithBG(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        width=896,
        height=384,
        interpolation="bicubic",
        placeholder_token="*",
        template="a photo of a {}",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.placeholder_token = placeholder_token

        self.image_paths = []
        self.image_paths += [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if is_image(file_path) and not 'bg' in file_path]

        self.image_paths = sorted(self.image_paths)

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.template = template

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process(self, image):
        img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):
        example = {}

        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = torch.tensor(placeholder_index)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        image = Image.open(self.image_paths[i % self.num_images])

        mask_path = self.image_paths[i % self.num_images].replace('.jpeg', '.png').replace('.jpg', '.png').replace('.JPEG', '.png')[:-4] + '_bg.png'
        mask = np.array(Image.open(mask_path))
        
        mask = np.where(mask > 0, 1, 0)
        
        
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)
        object_tensor = image_np * mask
        # covert numpy array to PIL Image
        Image.fromarray(object_tensor.astype('uint8')).save('temp.png')
        
        
        example["pixel_values"] = self.process(image_np)


        ref_object_tensor = Image.fromarray(object_tensor.astype('uint8')).resize((224, 224), resample=self.interpolation)
        ref_image_tenser = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)
        example["pixel_values_obj"] = self.get_tensor_clip()(ref_object_tensor)
        example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tenser)

        ref_seg_tensor = Image.fromarray(mask.astype('uint8') * 255)
        ref_seg_tensor = self.get_tensor_clip(normalize=False)(ref_seg_tensor)
        example["pixel_values_seg"] = torch.nn.functional.interpolate(ref_seg_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)

        return example


class OpenImagesDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        width=512,
        height=512,
        interpolation="bicubic",
        placeholder_token="*",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.placeholder_token = placeholder_token

        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])
        
        self.img_list = []
        imgtxt_file = os.path.join(data_root, 'mask_box.txt')
        # note read the bbox_path txt file using f
        with open(imgtxt_file, 'r') as f:
            for line in f:
                img_id = line.strip()
                self.img_list.append(img_id)

        self.num_images = len(self.img_list)

        print('{}: total {} images ...'.format(set, self.num_images))

        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_templates_small


    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process(self, image):
        img = np.array(image)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def obtain_text(self, add_caption, object_category=None):

        if object_category is None:
            placeholder_string = self.placeholder_token
        else:
            placeholder_string = object_category

        text = random.choice(self.templates).format(placeholder_string)
        text = add_caption + text[1:]

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        index = torch.tensor(placeholder_index)

        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return input_ids, index, text

    def __getitem__(self, i):
        
        example = {}
        input_ids, index, text = self.obtain_text('a')
        example["input_ids"] = input_ids
        example["index"] = index
        example["text"] = text

        img = self.img_list[i % self.num_images]
  

        img_path = os.path.join(self.data_root, 'data', img)
        img_p = Image.open(img_path).convert("RGB")
        img_p_np = np.array(img_p)
        
        # load mask
        mask = img.replace('.jpeg', '.png').replace('.jpg', '.png').replace('.JPEG', '.png')[:-4] + '_bg.png'
        mask_path = os.path.join(self.data_root, 'mask', mask)
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask > 0, 1, 0)
        
        image_tensor = img_p_np * mask
        image_tensor = image_tensor.astype('uint8')
        
        # save image_tensor to PIL image for debuging and checking
        Image.fromarray(image_tensor).save('temp.png')  
              
        # l_x, l_y, r_x, r_y,_ = bbox_sample
        # l_x, l_y, r_x, r_y = int(l_x), int(l_y), int(r_x), int(r_y)
        # note cut out the subject according to the bbox 
        # image_tensor = img_p_np[l_y:r_y,l_x:r_x,:]
        
        example["pixel_values"] = self.process(image_tensor)
        ref_image_tensor = self.random_trans(image=image_tensor)
        ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
        example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tensor)


        return example


class OpenImagesDatasetWithMask(OpenImagesDataset):
    def __init__(self,
        data_root,
        tokenizer,
        width=512,
        height=512,
        interpolation="bicubic",
        set="train",
        placeholder_token="*"):

        # super().__init__(data_root, tokenizer, size, interpolation, set, placeholder_token)
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.placeholder_token = placeholder_token
        self.set = set
        
        if self.width == self.height:
            self.size = self.width

        self.img_list = []
        imgtxt_file = os.path.join(data_root,'mask_box.txt')
        with open(imgtxt_file, 'r') as f:
            for line in f:
                img_id = line.strip()
                self.img_list.append(img_id)

        self.num_images = len(self.img_list)
            


        print('{}: total {} images ...'.format(set, self.num_images))

        self._length = self.num_images
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_templates_small


    def __len__(self):
        return self._length

    ## borrowed from custom diffusion
    def custom_aug(self, instance_image):
        
        instance_image = Image.fromarray(instance_image)
        #### apply augmentation and create a valid image regions mask ####
        if np.random.randint(0, 3) < 2:
            random_scale = np.random.randint(self.size // 3, self.size + 1)
        else:
            random_scale = np.random.randint(int(1.2 * self.size), int(1.4 * self.size))

        if random_scale % 2 == 1:
            random_scale += 1

        if random_scale < 0.6 * self.size:
            add_to_caption = np.random.choice(["a far away", "very small"])
            cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

            instance_image1 = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image1 = np.array(instance_image1).astype(np.uint8)
            instance_image1 = (instance_image1 / 127.5 - 1.0).astype(np.float32)

            instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
            instance_image[cx - random_scale // 2: cx + random_scale // 2,
            cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

            mask = np.zeros((self.size // 8, self.size // 8))
            mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1,
            (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.

        elif random_scale > self.size:
            add_to_caption = np.random.choice(["zoomed in", "close up"])
            cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

            instance_image = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            instance_image = instance_image[cx - self.size // 2: cx + self.size // 2,
                             cy - self.size // 2: cy + self.size // 2, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            add_to_caption = "a"
            if self.size is not None:
                instance_image = instance_image.resize((self.size, self.size), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))

        return torch.from_numpy(instance_image).permute(2, 0, 1), torch.from_numpy(mask[:, :, None]).permute(2, 0, 1), add_to_caption

    def aug_cv2(self, img):

        img_auged = np.array(img).copy()
        
        # resize and crop
        if random.choice([0, 1]) == 0:
            new_size = random.randint(224, 256)
            img_auged = cv2.resize(img_auged, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
            

            start_x, start_y = random.randint(0, new_size - 224), random.randint(0, new_size - 224)
            img_auged = img_auged[start_x:start_x + 224, start_y:start_y + 224, :]
            

        h, w = img_auged.shape[:2]
        # rotate
        if random.choice([0, 1]) == 0:
            # print('rotate')
            angle = random.randint(-30, 30)
            M = cv2.getRotationMatrix2D((112, 112), angle, 1)
            img_auged = cv2.warpAffine(img_auged, M, (w, h), flags=cv2.INTER_CUBIC)
            

        # translation
        if random.choice([0, 1]) == 0:
            trans_x = random.randint(-60, 60)
            trans_y = random.randint(-60, 60)
            H = np.float32([[1, 0, trans_x],
                            [0, 1, trans_y]])
            img_auged = cv2.warpAffine(img_auged, H, (w, h), flags=cv2.INTER_CUBIC)
           

        img_auged = Image.fromarray(img_auged)
       

        return img_auged


    def __getitem__(self, i):
        
        example = {}
        img = self.img_list[i % self.num_images]
        img_path = os.path.join(self.data_root,'data',img)
        img_p = Image.open(img_path).convert("RGB")
        img_p_np = np.array(img_p)
        
        mask = img.replace('.jpeg', '.png').replace('.jpg', '.png').replace('.JPEG', '.png')[:-4] + '_bg.png'
        mask_path = os.path.join(self.data_root, 'mask', mask)
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask > 0, 1, 0)
        
        # note by wy: seg the object region labeled by mask
        image_tensor = img_p_np * mask
        image_tensor = np.uint8(image_tensor)
        # note refer custom-diffusion paper
        # note augmentation for input image 
        augged_image, augged_mask, add_caption = self.custom_aug(image_tensor)
        input_ids, index, text = self.obtain_text(add_caption)

        example["pixel_values"] = augged_image
        example["mask_values"] = augged_mask
        example["input_ids"] = input_ids
        example["index"] = index
        example["text"] = text
        
        # note by wy: object is the same to image, since mask operation
        ref_object_tensor = cv2.resize(image_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
        ref_image_tenser = cv2.resize(image_tensor, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        ref_object_tensor = self.aug_cv2(ref_object_tensor.astype('uint8'))
        example["pixel_values_clip"] = self.get_tensor_clip()(Image.fromarray(ref_image_tenser))
        example["pixel_values_obj"] = self.get_tensor_clip()(ref_object_tensor)
        
        

        return example
