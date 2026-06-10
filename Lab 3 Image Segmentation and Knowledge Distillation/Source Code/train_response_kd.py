"""
train_response_kd_fixed.py

Response-based Knowledge Distillation training for LightweightUNetSeg student
with FCN-ResNet50 teacher on VOC2012.
Includes train/val loss plotting and val mIoU reporting.
"""

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchvision.transforms import InterpolationMode

from custom_model import LightweightUNetSeg

# ---------------------
# Response KD Loss
# ---------------------
def response_distillation_loss(student_logits, teacher_logits, labels,
                               alpha=1.0, beta=0.5, T=4.0, ignore_index=255):
    ce = F.cross_entropy(student_logits, labels, ignore_index=ignore_index)
    s_logp = F.log_softmax(student_logits / T, dim=1)
    t_p = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(s_logp, t_p, reduction='batchmean') * (T * T)
    return alpha * ce + beta * kd
    
# ---------------------
# VOC Dataloaders
# ---------------------
def voc_collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images,0), torch.stack(masks,0).long().squeeze(1)

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

# ---------------------
# Compute mIoU
# ---------------------
def compute_miou(preds, targets, num_classes=21):
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        inter = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if len(ious) > 0 else 0.0

def validate(student, val_loader, device):
    student.eval()
    total_loss, total_miou, count = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds, _ = student(imgs, return_feats=True)
            loss = F.cross_entropy(preds, targets, ignore_index=255)
            total_loss += loss.item()
            preds_cls = torch.argmax(preds, dim=1)
            total_miou += compute_miou(preds_cls.cpu(), targets.cpu())
            count += 1
    avg_loss = total_loss / max(count,1)
    avg_miou = total_miou / max(count,1)
    return avg_loss, avg_miou

# ---------------------
# Training
# ---------------------
def train_response_kd(voc_root, save_dir, epochs=30, batch_size=4, img_size=512,
                      alpha=1.0, beta=0.5, T=4.0, lr=3e-4):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_dataloaders(voc_root, batch_size, img_size)

    teacher = models.segmentation.fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1").to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False

    student = LightweightUNetSeg(num_classes=21, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = 1e9
    train_losses, val_losses, val_mious = [], [], []

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                t_out = teacher(imgs)['out']
            s_logits, _ = student(imgs, return_feats=True)
            loss = response_distillation_loss(s_logits, t_out, targets, alpha, beta, T)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step(epoch)
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss, val_miou = validate(student, val_loader, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_mious.append(val_miou)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val mIoU={val_miou:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), os.path.join(save_dir,'student_best_response.pth'))

    # Plot losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Response KD Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir,"loss_curve_response.png"))
    plt.close()

    # Plot Val mIoU
    plt.figure(figsize=(10,5))
    plt.plot(val_mious, label='Val mIoU', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir,"val_miou_curve_response.png"))
    plt.close()

    print("Training complete.")
    return student

# ---------------------
# Main
# ---------------------
if __name__=='__main__':
    data_root = '/content/data'
    save_dir = 'checkpoints'
    train_response_kd(voc_root=data_root, save_dir=save_dir, epochs=40, batch_size=4)