import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from custom_model import LightweightUNetSeg  # The model from previous step

# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss(pred, target, smooth=1e-6, ignore_index=255):
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
    mask = (target != ignore_index).unsqueeze(1)
    pred = pred * mask
    target_onehot = target_onehot * mask
    intersection = (pred * target_onehot).sum(dim=(0,2,3))
    union = pred.sum(dim=(0,2,3)) + target_onehot.sum(dim=(0,2,3))
    loss = 1 - ((2 * intersection + smooth) / (union + smooth))
    return loss.mean()

# -----------------------------
# Compute mIoU
# -----------------------------
def compute_iou(pred, target, num_classes=21):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if len(ious) > 0 else 0.0

# -----------------------------
# VOC Collate
# -----------------------------
def voc_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0).long().squeeze(1)
    return images, masks

# -----------------------------
# Dataloaders
# -----------------------------
def get_dataloaders(voc_root, batch_size=4, img_size=512):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomResizedCrop(img_size, scale=(0.5,1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((img_size,img_size), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    train_dataset = datasets.VOCSegmentation(voc_root, year="2012", image_set="train",
                                             download=False, transform=train_tf, target_transform=mask_tf)
    val_dataset = datasets.VOCSegmentation(voc_root, year="2012", image_set="val",
                                           download=False, transform=val_tf, target_transform=mask_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=voc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=voc_collate_fn)
    return train_loader, val_loader

# -----------------------------
# Training Function
# -----------------------------
def train_model(voc_root, num_epochs=50, lr=3e-4, batch_size=4, num_classes=21, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(voc_root, batch_size=batch_size)

    model = LightweightUNetSeg(num_classes=num_classes, pretrained=True).to(device)

    # Weighted CE for class imbalance
    class_weights = torch.ones(num_classes).to(device)
    class_weights[0] = 0.2  # background downweight
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    best_val_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_iou = 0, 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = ce_loss_fn(outputs, targets) + dice_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            running_iou += compute_iou(preds.cpu(), targets.cpu())
        scheduler.step(epoch + (0))  # CosineAnnealingWarmRestarts needs step(epoch)

        train_loss = running_loss / len(train_loader)
        train_iou = running_iou / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_iou = 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = ce_loss_fn(outputs, targets) + dice_loss(outputs, targets)
                val_loss += loss.item()
                preds = torch.argmax(outputs,1)
                val_iou += compute_iou(preds.cpu(), targets.cpu())

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, mIoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    # Plot curves
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss Curve")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_ious, label="Train mIoU")
    plt.plot(val_ious, label="Val mIoU")
    plt.title("mIoU Curve")
    plt.xlabel("Epoch"); plt.ylabel("mIoU"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"training_curves.png"))
    plt.close()
    print(f"✅ Training complete. Best Val mIoU: {best_val_iou:.4f}")
    return model

# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    voc_root = "/content/data"
    train_model(voc_root, num_epochs=50, lr=3e-4, batch_size=4)
