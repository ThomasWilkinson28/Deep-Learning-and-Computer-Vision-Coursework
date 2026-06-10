import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF
import math
import re
from PIL import Image, ImageFilter
import numpy as np

# ---------------- Custom Transforms ----------------
class Resize:
    def __init__(self, size):
        self.size = size  # (width, height)

    def __call__(self, image, uv):
        orig_w, orig_h = image.size
        new_w, new_h = self.size
        # Resize the image
        image = image.resize((new_w, new_h))
        x, y = uv
        # Scale the coordinates proportionally
        x = x * (new_w / orig_w)
        y = y * (new_h / orig_h)
        return image, (x, y)
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, uv):
        x, y = uv
        if random.random() < self.p:
            image = TF.hflip(image)
            x = image.width - x
        return image, (x, y)

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, uv):
        angle = random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle)

        #Rotate UV around image center
        cx, cy = image.width / 2, image.height / 2
        x, y = uv
        x -= cx
        y -= cy
        rad = -math.radians(angle)  # PIL rotates counter-clockwise
        new_x = x * math.cos(rad) - y * math.sin(rad) + cx
        new_y = x * math.sin(rad) + y * math.cos(rad) + cy
        return image, (new_x, new_y)
    
class RandomTranslation:
    def __init__(self, max_translate=0.1):
        """
        max_translate: float in [0,1], fraction of image dimension to shift.
        e.g., 0.1 means ±10% translation.
        """
        self.max_translate = max_translate

    def __call__(self, image, uv):
        width, height = image.size
        x_shift = random.uniform(-self.max_translate, self.max_translate) * width
        y_shift = random.uniform(-self.max_translate, self.max_translate) * height

        # Apply translation
        image = TF.affine(image, angle=0, translate=(x_shift, y_shift), scale=1.0, shear=0)

        # Adjust UV coordinates
        x, y = uv
        x += x_shift
        y += y_shift

        # Clamp coordinates to image boundaries
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return image, (x, y)

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        brightness, contrast, saturation, hue correspond to torchvision jitter ranges.
        """
        self.jitter = TF.adjust_brightness
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, uv):
        # Apply torchvision-like color jitter manually to preserve (image, uv) pairing
        image = TF.adjust_brightness(image, 1 + random.uniform(-self.brightness, self.brightness))
        image = TF.adjust_contrast(image, 1 + random.uniform(-self.contrast, self.contrast))
        image = TF.adjust_saturation(image, 1 + random.uniform(-self.saturation, self.saturation))
        image = TF.adjust_hue(image, random.uniform(-self.hue, self.hue))
        return image, uv

class GaussianBlur:
    def __init__(self, blur_prob=0.5, noise_std=0.02):
        """
        blur_prob: probability of applying blur (else apply noise)
        noise_std: standard deviation for Gaussian noise (0–1 range)
        """
        self.blur_prob = blur_prob
        self.noise_std = noise_std

    def __call__(self, image, uv):
        if random.random() < self.blur_prob:
            # Apply Gaussian blur
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius))
        else:
            # Apply Gaussian noise
            np_img = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.noise_std, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 1)
            image = Image.fromarray((np_img * 255).astype(np.uint8))
        return image, uv

# ---------------- Dataset ----------------
class SnoutDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Parse the label file
        self.data = []
        with open(label_file, 'r') as f:
            for line in f:
                # Example: beagle_145.jpg,"(198, 304)"
                match = re.match(r'([^,]+),"\((\d+),\s*(\d+)\)"', line.strip())
                if match:
                    img_name, x, y = match.groups()
                    self.data.append((img_name, (int(x), int(y))))
    
    # This will be later required by the DataLoader  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Get image and coordinates for specified index
        img_name, uv = self.data[idx]
        #Create a path for the image by adding it's name to the image file path
        img_path = os.path.join(self.img_dir, img_name)
        # Open the image ensuring it is RGB format
        image = Image.open(img_path).convert('RGB')

        for t in self.transform:
            image, uv = t(image, uv)
        
        #Convert to tensor
        image = TF.to_tensor(image)
        uv = torch.tensor(uv, dtype=torch.float32)
        return image, uv
    

