"""
train_feature_kd.py

Feature-based + Cross-Entropy Knowledge Distillation training script
for LightweightUNetSeg student and FCN-ResNet50 teacher on VOC2012.
Includes loss plotting and validation mIoU reporting.
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
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.transforms import InterpolationMode

from custom_model import LightweightUNetSeg


# ---------------------
# Feature KD Loss (Cosine + optional MSE)
# ---------------------
def feature_cosine_loss(f_s, f_t):
    if f_s.shape[-2:] != f_t.shape[-2:]:
        f_t = F.interpolate(f_t, size=f_s.shape[-2:], mode='bilinear', align_corners=True)
    f_s_n = F.normalize(f_s, dim=1)
    f_t_n = F.normalize(f_t, dim=1)
    cosine = 1.0 - (f_s_n * f_t_n).sum(dim=1).mean()
    mse = F.mse_loss(f_s, f_t)
    return 0.8 * cosine + 0.2 * mse


# ---------------------
# Adapter (match teacher & student channel dims)
# ---------------------
class FeatureAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------
# Teacher feature extractor (layer3 + layer4)
# ---------------------
def build_teacher_feature_extractor(teacher_model, return_layers=None):
    if return_layers is None:
        return_layers = {"layer3": "layer3", "layer4": "layer4"}
    backbone = teacher_model.backbone
    return IntermediateLayerGetter(backbone, return_layers)


# ---------------------
# Dataloaders (VOC2012)
# ---------------------
def voc_collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images, 0), torch.stack(masks, 0).long().squeeze(1)


def get_dataloaders(voc_root, batch_size=4, img_size=512):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.VOCSegmentation(
        voc_root, year="2012", image_set="train",
        download=False, transform=train_tf, target_transform=mask_tf
    )
    val_dataset = datasets.VOCSegmentation(
        voc_root, year="2012", image_set="val",
        download=False, transform=val_tf, target_transform=mask_tf
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=voc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=voc_collate_fn)
    return train_loader, val_loader


# ---------------------
# mIoU computation
# ---------------------
def compute_mIoU(pred, target, num_classes=21, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    mask = target != ignore_index
    pred, target = pred[mask], target[mask]
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


# ---------------------
# Validation
# ---------------------
def validate_feature_kd(student, teacher_feat_extractor, adapter_layer3, adapter_layer4, val_loader, device, alpha=1.0, beta=1.0):
    student.eval()
    total_loss, total_miou, count = 0.0, 0.0, 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            t_feats = teacher_feat_extractor(imgs)
            s_logits, s_feats = student(imgs, return_feats=True)

            # CE loss
            ce_loss = F.cross_entropy(s_logits, targets, ignore_index=255)

            # Feature KD
            mid_loss = feature_cosine_loss(s_feats['mid'], adapter_layer3(t_feats['layer3']))
            high_loss = feature_cosine_loss(s_feats['high'], adapter_layer4(t_feats['layer4']))
            feat_loss = 0.3 * mid_loss + 0.7 * high_loss

            loss = alpha * ce_loss + beta * feat_loss
            total_loss += loss.item()

            preds = torch.argmax(s_logits, dim=1)
            total_miou += compute_mIoU(preds.cpu(), targets.cpu())
            count += 1

    return total_loss / max(count, 1), total_miou / max(count, 1)


# ---------------------
# Training (CE + Feature KD)
# ---------------------
def train_feature_kd(voc_root, save_dir, epochs=40, batch_size=4, img_size=512,
                     lr=3e-4, alpha=1.0, beta=0.5):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_dataloaders(voc_root, batch_size, img_size)

    # Teacher
    teacher = models.segmentation.fcn_resnet50(
        weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1"
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_feat_extractor = build_teacher_feature_extractor(teacher).to(device)

    # Student
    student = LightweightUNetSeg(num_classes=21, pretrained=True).to(device)

    # Feature adapters
    adapter_layer3 = FeatureAdapter(1024, student.mid_ch).to(device)
    adapter_layer4 = FeatureAdapter(2048, student.high_ch).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) +
        list(adapter_layer3.parameters()) +
        list(adapter_layer4.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_miou = 0.0
    train_losses, val_losses, val_mious = [], [], []

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Teacher features
            with torch.no_grad():
                t_feats = teacher_feat_extractor(imgs)

            # Student forward
            s_logits, s_feats = student(imgs, return_feats=True)

            # 1️⃣ Cross-entropy segmentation loss
            ce_loss = F.cross_entropy(s_logits, targets, ignore_index=255)

            # 2️⃣ Feature distillation loss (weighted cosine)
            mid_loss = feature_cosine_loss(s_feats['mid'], adapter_layer3(t_feats['layer3']))
            high_loss = feature_cosine_loss(s_feats['high'], adapter_layer4(t_feats['layer4']))
            feat_loss = 0.3 * mid_loss + 0.7 * high_loss

            # 3️⃣ Total loss
            loss = alpha * ce_loss + beta * feat_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step(epoch)
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        avg_val_loss, val_miou = validate_feature_kd(
            student, teacher_feat_extractor, adapter_layer3, adapter_layer4, val_loader, device, alpha, beta
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_mious.append(val_miou)

        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val mIoU={val_miou:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(student.state_dict(), os.path.join(save_dir, 'student_best_kd_ce.pth'))
            print(f"✅ Saved best model (mIoU={best_val_miou:.4f})")

    # Plot Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Feature KD + CE Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve_kd_ce.png"))
    plt.close()

    # Plot mIoU
    plt.figure(figsize=(8, 5))
    plt.plot(val_mious, label='Val mIoU', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "val_miou_kd_ce.png"))
    plt.close()

    print(f"✅ Training complete. Best Val mIoU: {best_val_miou:.4f}")
    return student


# ---------------------
# Main
# ---------------------
if __name__ == '__main__':
    data_root = '/content/data'
    save_dir = 'checkpoints_kd_ce'
    train_feature_kd(
        voc_root=data_root,
        save_dir=save_dir,
        epochs=50,
        batch_size=4,
        img_size=512,
        alpha=1.0,   # CE loss weight
        beta=0.5     # Feature KD loss weight
    )