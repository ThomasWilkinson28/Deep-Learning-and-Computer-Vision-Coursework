import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define the standard CLIP image normalization
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def get_transforms(split='no_augment'):
    if split == 'augment':
        return transforms.Compose([
            # 1. Spatial: Randomly crop and resize
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            # 2. Spatial: Horizontal flipping
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. Spatial: Slight rotation
            transforms.RandomRotation(degrees=15),
            # 4. Color Jitter: Randomly change brightness, contrast, saturation, and hue
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # 5. Grayscale: Randomly drop color information to force structure learning
            transforms.RandomGrayscale(p=0.2),
            # -------------------------------------

            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    else:
        # (Keep validation transforms the same)
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

class CocoClipDataset(Dataset):
    """
    A PyTorch Dataset that implements the Sampling Strategy.
    
    Expected Cache Format: 
    A list of tuples: [(image_filename, tensor_of_all_captions), ...]
    where tensor_of_all_captions has shape (Num_Captions, 512).
    """
    def __init__(self, image_dir, data_cache_file, transform=None, mode = 'train'):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        print(f"Loading data from cache: {data_cache_file}")
        # Load the data: List of (filename, tensor_stack)
        self.data = torch.load(data_cache_file)
        print(f"Loaded {len(self.data)} items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Shape (3, 224, 224)
            text_embedding (Tensor): Shape (512,) -> One randomly selected caption
        """
        # 1. Unpack data
        # caption_group is a tensor of shape (N, 512), usually N=5
        image_filename, caption_group = self.data[idx]
        
        # 2. STRATEGY: Random Sampling
        # We pick ONE caption index randomly from the available options.
        # This acts as text augmentation across epochs.
        num_captions = caption_group.shape[0]
        
        if num_captions > 0:
            if self.mode == 'train':
                # Random sampling for training
                selected_idx = torch.randint(0, num_captions, (1,)).item()
            else:
                # Deterministic sampling for validation/testing (Always take the first one)
                selected_idx = 0
                
            selected_embedding = caption_group[selected_idx]
        else:
            # Fallback (should not happen in COCO)
            selected_embedding = torch.zeros(512)

        # 3. Load and process the image
        image_path = os.path.join(self.image_dir, image_filename)
        
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (IOError, FileNotFoundError) as e:
            print(f"Warning: Could not load image {image_path}. Skipping.")
            # Recursive fallback to keep batch size constant
            return self.__getitem__((idx + 1) % len(self))
            
        return image, selected_embedding



# Varifying dataset script with training set plotting augmented images and their associated captions:
if __name__ == '__main__':
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from pycocotools.coco import COCO

    # --- Constants (Must match your training script) ---
    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
    CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])

    # Paths
    COCO_ROOT = 'coco2014' # Adjust if needed
    TRAIN_IMAGE_DIR = os.path.join(COCO_ROOT, 'train2014')
    TRAIN_CACHE_FILE = 'cache/coco_train_embeddings5to1.pt' # Ensure this path is correct
    ANNOTATION_FILE = os.path.join(COCO_ROOT, 'annotations/captions_train2014.json')

    def denormalize_image(tensor):
        """
        Reverses the CLIP normalization for visualization.
        Input: Tensor (3, 224, 224)
        Output: Numpy Array (224, 224, 3) range [0, 1]
        """
        # Clone to avoid modifying the original tensor
        img = tensor.clone().detach().cpu()
        # 1. Rearrange dims: (C, H, W) -> (H, W, C)
        img = img.permute(1, 2, 0).numpy()
        # 2. Denormalize: image = (image * std) + mean
        img = (img * CLIP_STD) + CLIP_MEAN
        # 3. Clip values to valid [0, 1] range
        img = np.clip(img, 0, 1)
        return img

    def verify_dataset_integrity(num_samples=5):
        print("--- 1. Loading COCO Annotations (for text lookup) ---")
        coco = COCO(ANNOTATION_FILE)
        
        # Create a helper map: Filename -> Image ID
        # This is needed because your dataset has filenames, but COCO works by IDs
        print("Building filename index...")
        filename_to_id = {img_info['file_name']: img_id for img_id, img_info in coco.imgs.items()}

        print("\n--- 2. Initializing Dataset ---")
        # We use 'augment' to verify that transformations are working correctly
        dataset = CocoClipDataset(
            image_dir=TRAIN_IMAGE_DIR, 
            data_cache_file=TRAIN_CACHE_FILE, 
            transform=get_transforms('augment'), # Check augmented version
            mode='train'
        )

        print(f"\n--- 3. Sampling {num_samples} items ---")
        indices = random.sample(range(len(dataset)), num_samples)

        plt.figure(figsize=(16, 6 * num_samples))

        for i, idx in enumerate(indices):
            # A. Get the transformed data from the Dataset Class
            # This calls __getitem__ and triggers the transforms
            image_tensor, embedding_tensor = dataset[idx]

            # B. Get the Metadata (Filename) directly from internal storage
            # access dataset.data[idx] -> returns (filename, embedding_stack)
            filename, _ = dataset.data[idx]

            # C. Retrieve Original Captions via COCO API
            if filename in filename_to_id:
                img_id = filename_to_id[filename]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                captions = [ann['caption'] for ann in anns]
            else:
                captions = ["Error: Could not find caption in JSON for this file."]

            # D. Prepare Image for Display
            display_img = denormalize_image(image_tensor)

            # E. Plotting
            # Image on the Left
            ax_img = plt.subplot(num_samples, 2, 2 * i + 1)
            ax_img.imshow(display_img)
            ax_img.set_title(f"Dataset Index: {idx}\nShape: {tuple(image_tensor.shape)}")
            ax_img.axis('off')

            # Text on the Right
            ax_txt = plt.subplot(num_samples, 2, 2 * i + 2)
            
            text_content = f"Filename: {filename}\n"
            text_content += f"Embedding Shape: {tuple(embedding_tensor.shape)}\n"
            text_content += "-" * 40 + "\nAssociated Captions (from JSON):\n"
            
            for cap in captions:
                text_content += f"• {cap}\n"

            ax_txt.text(0.05, 0.5, text_content, 
                    fontsize=12, 
                    verticalalignment='center', 
                    fontfamily='monospace',
                    wrap=True)
            ax_txt.axis('off')

        plt.tight_layout()
        plt.show()
        print("Verification Complete.")

    if __name__ == "__main__":
        if os.path.exists(TRAIN_CACHE_FILE) and os.path.exists(ANNOTATION_FILE):
            verify_dataset_integrity(num_samples=5)
        else:
            print(f"Error: Files not found.")
            print(f"Looked for cache at: {TRAIN_CACHE_FILE}")
            print(f"Looked for annotations at: {ANNOTATION_FILE}")
     