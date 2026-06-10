import os
import torch
from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO
from tqdm import tqdm

# Define the model ID
MODEL_ID = "openai/clip-vit-base-patch32"

def load_model_and_processor(device):
    """Loads the CLIP model and processor."""
    print(f"Loading model: {MODEL_ID}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()  # Set model to evaluation mode
    return model, processor

def process_and_cache_split(split, coco_root, out_dir, model, processor, device):
    """
    Processes a single split (train/val) of the COCO dataset.
    Grouping Strategy: Groups all captions for a single image into one tensor.
    Structure: [(image_filename, tensor_of_shape_N_x_512), ...]
    """
    if split == 'train':
        caption_file = os.path.join(coco_root, 'annotations', 'captions_train2014.json')
        image_dir = os.path.join(coco_root, 'train2014')
    else:
        caption_file = os.path.join(coco_root, 'annotations', 'captions_val2014.json')
        image_dir = os.path.join(coco_root, 'val2014')
        
    cache_file = os.path.join(out_dir, f"coco_{split}_embeddings5to1.pt")
    
    if os.path.exists(cache_file):
        print(f"Cache file found at {cache_file}. Skipping preprocessing.")
        return

    print(f"Processing split: {split}")
    print(f"Loading annotations from: {caption_file}")
    
    coco = COCO(caption_file)
    # UPDATED: Get Image IDs instead of Annotation IDs
    img_ids = coco.getImgIds()
    
    print(f"Found {len(img_ids)} images.")
    
    all_data = []
    
    # Loop through unique images
    for img_id in tqdm(img_ids, desc=f"Encoding {split} images"):
        
        # 1. Get image info to find the filename
        img_info = coco.loadImgs(img_id)[0]
        image_filename = img_info['file_name']
        
        # 2. Verify image file exists before processing text
        if not os.path.exists(os.path.join(image_dir, image_filename)):
            continue

        # 3. Get all captions for this specific image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Extract the text strings (usually 5 of them)
        captions_list = [ann['caption'] for ann in anns]
        
        # 4. Preprocess and encode all captions for this image at once
        if len(captions_list) > 0:
            try:
                # Tokenize list of strings
                inputs = processor(
                    text=captions_list, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True
                ).to(device)
                
                with torch.no_grad():
                    # Get the text features
                    # Output shape will be (Num_Captions, 512) -> e.g., (5, 512)
                    text_features = model.get_text_features(**inputs)
                
                # Store filename and the resulting stacked embeddings (move to CPU)
                # We save it as one tensor containing all caption variants
                all_data.append((image_filename, text_features.cpu()))
                
            except Exception as e:
                print(f"Warning: Skipping image {image_filename} due to error: {e}")
                continue

    print(f"Processed {len(all_data)} valid (image, caption_group) pairs.")
    
    # Save the processed data to the cache file
    os.makedirs(out_dir, exist_ok=True)
    torch.save(all_data, cache_file)
    print(f"Saved cached embeddings to {cache_file}")

def main():
    # Update this path to where your COCO dataset is located
    coco_root = 'coco2014' 
    out_dir = './cache'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, processor = load_model_and_processor(device)
    
    process_and_cache_split('train', coco_root, out_dir, model, processor, device)
    process_and_cache_split('val', coco_root, out_dir, model, processor, device)
    
    print("All processing complete.")

if __name__ == "__main__":
    main()