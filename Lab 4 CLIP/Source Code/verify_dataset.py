import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import textwrap

"This script is used to verify that the COCO dataset and captions match"


def verify_data(coco_root, split='train', num_samples=5):
    """
    Loads and displays a few random image-caption pairs from the dataset.
    """
    if split == 'train':
        caption_file = os.path.join(coco_root, 'annotations', 'captions_train2014.json')
        image_dir = os.path.join(coco_root, 'train2014')
    else:
        caption_file = os.path.join(coco_root, 'annotations', 'captions_val2014.json')
        image_dir = os.path.join(coco_root, 'val2014')

    print(f"Loading annotations from: {caption_file}")
    coco = COCO(caption_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Sample random image IDs
    random_img_ids = random.sample(img_ids, num_samples)
    
    print(f"Displaying {num_samples} random samples...")
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, img_id in enumerate(random_img_ids):
        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_dir, img_info['file_name'])
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found {image_path}")
            continue
            
        # Load annotations (captions)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]
        
        # Format captions for display
        caption_text = f"Image ID: {img_id} ({img_info['file_name']})\n\n"
        caption_text += "\n".join([f"- {cap}" for cap in captions])
        
        # Plot
        ax = plt.subplot(num_samples, 2, 2*i + 1)
        ax.imshow(image)
        ax.axis('off')
        
        ax_text = plt.subplot(num_samples, 2, 2*i + 2)
        ax_text.text(0, 0.5, textwrap.fill(caption_text, 90), 
                     verticalalignment='center', 
                     fontsize=10, 
                     wrap=True)
        ax_text.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- IMPORTANT ---
    COCO_ROOT = 'coco2014/' 
    
    if not os.path.exists(COCO_ROOT):
        print(f"Error: COCO_ROOT path not found: {COCO_ROOT}")
        print("Please download the COCO 2014 dataset and update the COCO_ROOT variable.")
    else:
        verify_data(COCO_ROOT, split='train', num_samples=3)