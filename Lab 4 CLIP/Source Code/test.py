import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel
import argparse

# Import your custom modules
# Make sure DatasetSetupSampling5to1 contains the UPDATED class above
from DatasetSetup import CocoClipDataset, get_transforms
from resnet50_model import ImageEncoder, ImageEncoderModified

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_IMAGE_DIR = 'coco2014/val2014' 
VAL_EMBEDS_PATH = 'cache/coco_val_embeddings5to1.pt' 
HF_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def load_text_encoder():
    """Loads the FULL HuggingFace CLIP Model to access .get_text_features()"""
    print("Loading HF CLIP Model for visualization...")
    tokenizer = CLIPTokenizer.from_pretrained(HF_CLIP_MODEL_NAME)
    # LOAD THE FULL MODEL, NOT JUST .text_model
    full_clip_model = CLIPModel.from_pretrained(HF_CLIP_MODEL_NAME).to(DEVICE)
    full_clip_model.eval()
    return tokenizer, full_clip_model


def load_model_and_checkpoint(model_path, model_name):
    """Loads model and the learned Logit Scale."""
    print(f"Loading model from {model_path}...")
    if model_name == 'normal_model':
        model = ImageEncoder(embedding_dim=512).to(DEVICE)
    else:
        model = ImageEncoderModified(embedding_dim=512).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Load Model Weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from Epoch {checkpoint.get('epoch', '?')} with Loss {checkpoint.get('loss', '?')}")
        
        # Load Logit Scale (Needed for Zero-Shot probability viz)
        logit_scale = checkpoint.get('logit_scale', torch.tensor(np.log(1/0.07))).to(DEVICE)
    else:
        model.load_state_dict(checkpoint)
        logit_scale = torch.tensor(np.log(1/0.07)).to(DEVICE) # Default if missing
        
    model.eval()
    return model, logit_scale

def load_text_encoder():
    """
    Loads the FULL HuggingFace CLIP Model.
    We need the full model to access .get_text_features() which includes the projection layer.
    """
    print("Loading HF CLIP Model for visualization...")
    tokenizer = CLIPTokenizer.from_pretrained(HF_CLIP_MODEL_NAME)
    
    # LOAD THE FULL MODEL (Remove .text_model at the end)
    full_clip_model = CLIPModel.from_pretrained(HF_CLIP_MODEL_NAME).to(DEVICE)
    full_clip_model.eval()
    
    return tokenizer, full_clip_model

def get_all_embeddings(model, loader):
    """
    Runs inference on the validation set.
    """
    image_embeds = []
    text_embeds = []
    
    print("Generating embeddings for validation set...")
    with torch.no_grad():
        for images, cached_text_embeds in tqdm(loader):
            images = images.to(DEVICE)
            cached_text_embeds = cached_text_embeds.to(DEVICE)
            
            # 1. Image Embeddings
            img_out = model(images)
            
            # 2. Normalize (Recall@K relies on Cosine Similarity)
            img_out = F.normalize(img_out, p=2, dim=1)
            txt_out = F.normalize(cached_text_embeds, p=2, dim=1)
            
            image_embeds.append(img_out.cpu())
            text_embeds.append(txt_out.cpu())
            
    return torch.cat(image_embeds), torch.cat(text_embeds)

def calculate_recall_at_k(image_embeds, text_embeds, k_values=[1, 5, 10]):
    # Matrix Multiplication (No logit scale needed for ranking)
    # shape: (Num_Images, Num_Images)
    logits = torch.matmul(image_embeds.float(), text_embeds.float().T)
    
    num_samples = logits.shape[0]
    # The target for Image_i is Text_i (Diagonal)
    targets = torch.arange(num_samples).to(logits.device)
    
    results = {}
    
    # Image to Text
    _, indices = torch.topk(logits, k=max(k_values), dim=1)
    for k in k_values:
        correct = indices[:, :k].eq(targets.unsqueeze(1).expand(-1, k))
        results[f'Image-to-Text R@{k}'] = (correct.float().sum() / num_samples).item() * 100

    # Text to Image
    _, indices = torch.topk(logits.T, k=max(k_values), dim=1)
    for k in k_values:
        correct = indices[:, :k].eq(targets.unsqueeze(1).expand(-1, k))
        results[f'Text-to-Image R@{k}'] = (correct.float().sum() / num_samples).item() * 100
        
    return results

def run_visualizations(model, logit_scale, image_embeds, dataset, tokenizer, text_encoder, image_dir):
    image_embeds = image_embeds.to(DEVICE) 
    
    # Exponentiate the logit scale to get the actual scalar (e.g., 14.3)
    logit_scale_scalar = logit_scale.exp() 

    def get_image_path(idx):
        if isinstance(dataset, torch.utils.data.Subset):
            real_idx = dataset.indices[idx]
            filename, _ = dataset.dataset.data[real_idx]
        else:
            filename, _ = dataset.data[idx]
        return os.path.join(image_dir, filename)

    # --- Task 1: Text-to-Image Retrieval ---
    queries = ["building", "group of animals", "swimming", "parking lot"]
    fig, axes = plt.subplots(len(queries), 6, figsize=(20, 3 * len(queries)))
    
    print("\nGenerating Retrieval Visualization...")
    for i, query in enumerate(queries):
        inputs = tokenizer([query], padding=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            # FIX: Use get_text_features to include the projection layer!
            text_emb = text_encoder.get_text_features(**inputs)
            # Normalize
            text_emb = F.normalize(text_emb, p=2, dim=1)
            
        # Similarity = Dot Product (Ranking doesn't need scale)
        sims = (image_embeds @ text_emb.T).squeeze()
        topk_vals, topk_idxs = torch.topk(sims, k=5)
        
        axes[i, 0].text(0.5, 0.5, query, ha='center', va='center', fontsize=12, wrap=True)
        axes[i, 0].axis('off')
        
        for j, idx in enumerate(topk_idxs):
            img_idx = idx.item()
            try:
                img = Image.open(get_image_path(img_idx)).convert("RGB")
                axes[i, j+1].imshow(img)
                axes[i, j+1].set_title(f"{topk_vals[j]:.2f}")
            except Exception as e:
                print(e)
            axes[i, j+1].axis('off')
            
    plt.tight_layout()
    plt.savefig("viz_retrieval.png")

    # --- Task 2: Zero-Shot Classification ---
    rand_idx = np.random.randint(0, len(image_embeds))
    classes = ['a person', 'an animal', 'a landscape', 'food', 'a vehicle']
    
    text_inputs = tokenizer(classes, padding=True, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # FIX: Use get_text_features here too
        class_embeds = text_encoder.get_text_features(**text_inputs)
        class_embeds = F.normalize(class_embeds, p=2, dim=1)
        
        img_emb = image_embeds[rand_idx].unsqueeze(0)
        
        # Calculate logits using the learned scale
        logits = (img_emb @ class_embeds.T) * logit_scale_scalar
        
        # Softmax and convert to numpy safely
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    try:
        ax1.imshow(Image.open(get_image_path(rand_idx)).convert("RGB"))
        ax1.set_title("Input Image")
    except Exception as e:
        print(f"Error loading classification image: {e}")
        
    ax1.axis('off')
    
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_title('Zero-Shot Classification')
    
    plt.tight_layout()
    plt.savefig("viz_classification.png")
    print("Saved 'viz_classification.png'")

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-a', metavar='state', type=str, help='with or without augmentation')
    argParser.add_argument('-m', metavar='state', type=str, help='model')
   
    args = argParser.parse_args()

    if args.a != None:
        augmentation = args.a
    if args.m != None:
        model_name = args.m

    if augmentation == 'no_augment' and model_name == 'normal_model':
        MODEL_PATH = f'./clip_resnet50_NoModifications.pt' 
    
    elif augmentation == 'augment' and model_name == 'normal_model':
        MODEL_PATH = f'./clip_resnet50_Augmentation.pt' 
    
    elif augmentation == 'no_augment' and model_name == 'modified_model':
        MODEL_PATH = f'./clip_resnet50_ModifiedModel.pt'
    
    else:
        MODEL_PATH = f'./clip_resnet50_BothModifications.pt'  

    torch.cuda.empty_cache()
    
    # 1. Load Data with mode='val' (Deterministic)
    val_dataset = CocoClipDataset(
        image_dir=VAL_IMAGE_DIR,
        data_cache_file=VAL_EMBEDS_PATH, 
        transform=get_transforms('no_augment'),
        mode='val' 
    )
    
    # Subset for speed if needed (e.g. 5000 samples)
    VAL_SUBSET_SIZE = 5000 
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(val_dataset), generator=generator)[:VAL_SUBSET_SIZE]
    val_subset = torch.utils.data.Subset(val_dataset, indices)
    
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Load Model & Scale
    model, logit_scale = load_model_and_checkpoint(MODEL_PATH, model_name)
    
    # 3. Calculate Metrics
    all_image_embeds, all_text_embeds = get_all_embeddings(model, val_loader)
    metrics = calculate_recall_at_k(all_image_embeds, all_text_embeds)
    
    print("\n" + "-" * 30)
    for k, v in metrics.items():
        print(f"{k:<20} | {v:.2f}%")
    print("-" * 30)
    
    # 4. Visualize
    tokenizer, text_encoder = load_text_encoder()
    run_visualizations(model, logit_scale, all_image_embeds, val_subset, tokenizer, text_encoder, VAL_IMAGE_DIR)

if __name__ == "__main__":
    main()