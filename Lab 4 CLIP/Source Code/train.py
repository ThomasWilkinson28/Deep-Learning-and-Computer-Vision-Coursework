import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Import custom dataset and model
from DatasetSetup import CocoClipDataset, get_transforms
from resnet50_model import ImageEncoder, ImageEncoderModified

# --- 1. InfoNCE Loss Definition ---

def info_nce_loss(image_embeddings, text_embeddings, logit_scale):
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Cosine similarity
    logits = (image_embeddings @ text_embeddings.T) * logit_scale.exp()
    
    # Labels are 0, 1, 2... (Diagonal)
    labels = torch.arange(len(logits), device=logits.device)
    
    loss_images = F.cross_entropy(logits, labels)
    loss_texts = F.cross_entropy(logits.T, labels)
    
    loss = (loss_images + loss_texts) / 2
    return loss

# --- 2. Configuration & Hyperparameters ---

# Paths and params for colab
'''
COCO_ROOT = '/content/'
TRAIN_IMAGE_DIR = os.path.join(COCO_ROOT, 'train2014')
VAL_IMAGE_DIR = os.path.join(COCO_ROOT, 'val2014')
TRAIN_CACHE_FILE = '/content/coco_train_embeddings5to1.pt'
VAL_CACHE_FILE = '/content/coco_val_embeddings5to1.pt'
MODEL_SAVE_PATH = '/content/clip_resnet50_best.pt'

BATCH_SIZE = 734
NUM_WORKERS = 12

'''

#Paths and params for PC
COCO_ROOT = 'coco2014'
TRAIN_IMAGE_DIR = os.path.join(COCO_ROOT, 'train2014')
VAL_IMAGE_DIR = os.path.join(COCO_ROOT, 'val2014')
TRAIN_CACHE_FILE = 'cache/coco_train_embeddings5to1.pt'
VAL_CACHE_FILE = 'cache/coco_val_embeddings5to1.pt'
MODEL_SAVE_PATH = 'clip_resnet50_best.pt'

# Hyperparameters
NUM_EPOCHS = 60
BATCH_SIZE = 32  # Keep this as high as your GPU memory allows!
LEARNING_RATE = 5e-4 
WEIGHT_DECAY = 0.05  
EMBEDDING_DIM = 512
NUM_WORKERS = 0 #Set to whatever CPU allows
MAX_GRAD_NORM = 1.0 # Gradient Clipping Threshold

# --- 3. Training Function (Updated) ---

def train_one_epoch(model, loader, optimizer, scaler, logit_scale, device):
    model.train() 
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        images, text_embeds = batch
        images = images.to(device, non_blocking=True)
        text_embeds = text_embeds.to(device, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            image_embeds = model(images)
            loss = info_nce_loss(image_embeds, text_embeds, logit_scale)
        
        optimizer.zero_grad()
        
        # 1. Scale Loss & Backward
        scaler.scale(loss).backward()
        
        # 2. Unscale Gradients (CRITICAL STEP for Clipping with AMP)
        scaler.unscale_(optimizer)
        
        # 3. Clip Gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        
        # 4. Step Optimizer
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate_one_epoch(model, loader, logit_scale, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images, text_embeds = batch
            images = images.to(device, non_blocking=True)
            text_embeds = text_embeds.to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                image_embeds = model(images)
                loss = info_nce_loss(image_embeds, text_embeds, logit_scale)

            total_loss += loss.item()
            
    return total_loss / len(loader)

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    plt.close() 

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-a', metavar='state', type=str, help='with or without augmentation')
    argParser.add_argument('-m', metavar='state', type=str, help='model')
   
    args = argParser.parse_args()

    if args.a != None:
        augmentation = args.a
    if args.m != None:
        model_name = args.m
    
    torch.backends.cudnn.benchmark = True
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Datasets ---
    print("Loading datasets...")
    try:
        if augmentation == 'augment':
            print('Training with data augmentation')
            train_dataset = CocoClipDataset(image_dir=TRAIN_IMAGE_DIR, data_cache_file=TRAIN_CACHE_FILE, transform=get_transforms('augment'), mode = 'train') 
        else:
            print('Training without data augmentation')
            train_dataset = CocoClipDataset(image_dir=TRAIN_IMAGE_DIR, data_cache_file=TRAIN_CACHE_FILE, transform=get_transforms('no_augment'), mode = 'train')
        
        val_dataset = CocoClipDataset(
            image_dir=VAL_IMAGE_DIR,
            data_cache_file=VAL_CACHE_FILE,
            transform=get_transforms('no_augment'),
            mode = 'val'
        )
    except FileNotFoundError as e:
        print(f"Error: Cache file not found. {e.filename}")
        print("Please download preprocessed train text embeddings .pt file from the one drive link in 'TrainTextEmbeddingsLink.txt' and add the file to the cache folder.")
        exit()
    
    VAL_SUBSET_SIZE = 15000 
    if VAL_SUBSET_SIZE is not None and VAL_SUBSET_SIZE < len(val_dataset):
        print(f"Subsetting validation set to {VAL_SUBSET_SIZE} images...")
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(val_dataset), generator=generator)[:VAL_SUBSET_SIZE]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Loaded {len(train_dataset)} train samples and {len(val_dataset)} val samples.")

    # --- Initialize Model ---
    print("Initializing model...")
    if model_name == 'normal_model':
        print('Basic Model')
        model = ImageEncoder(embedding_dim=EMBEDDING_DIM).to(device)
    else:
        print('Modified Model')
        model = ImageEncoderModified(embedding_dim=EMBEDDING_DIM).to(device)

    # Logit Scale
    initial_log_temp = np.log(1 / 0.07)
    logit_scale = nn.Parameter(torch.tensor(initial_log_temp, device=device))

    # --- IMPROVEMENT: AdamW Optimizer ---
    backbone_params = model.model.parameters() # The ResNet weights
    head_params = model.projection.parameters() # The new Linear/BN layers
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5}, # Low LR for pretrained weights to preserve features
        {'params': head_params, 'lr': 5e-4},     # Higher LR for new layers to learn quickly
        {'params': [logit_scale], 'lr': 5e-4}
    ], weight_decay=WEIGHT_DECAY)
    
    # Note: When using parameter groups, the scheduler will update all of them proportionally
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    # --- Loop ---
    readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"Start Time: {readable_time}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, logit_scale, device)
        scheduler.step()
        val_loss = validate_one_epoch(model, val_loader, logit_scale, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'logit_scale': logit_scale, 
                'loss': best_val_loss,
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            plot_loss_curves(train_losses, val_losses)
            
        epoch_end_time = time.time()
        print(f"Epoch Time: {(epoch_end_time - epoch_start_time) / 60:.2f} minutes")

    total_training_time = time.time() - start_time
    print("\nTraining complete.")
    print(f"Total Training Time: {total_training_time / 60:.2f} minutes")
    print(f"Best Validation Loss: {best_val_loss:.4f}")