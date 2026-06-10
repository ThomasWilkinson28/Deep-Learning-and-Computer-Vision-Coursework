"""
Step One: Test Pretrained FCN-ResNet50 on Local PASCAL VOC 2012 Dataset
With visualization and pixel accuracy
-----------------------------------------------------------------------
"""

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------
# Utility: VOC colormap
# ------------------------------
def voc_colormap(n=21):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r, g, b = 0, 0, 0
        cid = i
        for j in range(8):
            r |= (bitget(cid, 0) << 7 - j)
            g |= (bitget(cid, 1) << 7 - j)
            b |= (bitget(cid, 2) << 7 - j)
            cid >>= 3
        cmap[i] = [r, g, b]
    return cmap

# ------------------------------
# Metrics computation
# ------------------------------
def compute_metrics(preds, gts, num_classes=21):
    ious, total_correct, total_label = [], 0, 0
    for pred, gt in zip(preds, gts):
        mask = gt != 255
        pred, gt = pred[mask], gt[mask]
        total_correct += (pred == gt).sum()
        total_label += mask.sum()
        for cls in range(num_classes):
            p, g = pred == cls, gt == cls
            inter = (p & g).sum()
            union = p.sum() + g.sum() - inter
            if union > 0:
                ious.append(inter / union)
    acc = total_correct / (total_label + 1e-10)
    miou = np.mean(ious)
    return miou, acc

# ------------------------------
# Visualization
# ------------------------------
def visualize_sample(image, pred, gt, cmap):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    gt_color = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pred_color = np.zeros_like(gt_color)

    ignore_mask = (gt == 255)
    gt_masked = np.where(ignore_mask, 0, gt)
    pred_masked = np.where(ignore_mask, 0, pred)

    gt_color[~ignore_mask] = cmap[gt_masked[~ignore_mask]]
    pred_color[~ignore_mask] = cmap[pred_masked[~ignore_mask]]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Input")
    ax[1].imshow(pred_color)
    ax[1].set_title("Prediction")
    ax[2].imshow(gt_color)
    ax[2].set_title("Ground Truth")

    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Evaluation
# ------------------------------
def evaluate(model, dataloader, device, num_classes=21, visualize=False):
    model.eval()
    preds_all, gts_all = [], []
    cmap = voc_colormap(num_classes)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.squeeze(0).long().to(device)

            outputs = model(images)["out"]
            preds = torch.argmax(outputs[0], dim=0)

            preds_all.append(preds.cpu().numpy())
            gts_all.append(targets.cpu().numpy())

            if visualize and i < 20:
                visualize_sample(images[0].cpu(), preds.cpu().numpy(), targets.cpu().numpy(), cmap)

    miou, acc = compute_metrics(preds_all, gts_all, num_classes)
    print(f"\n✅ Evaluation Results:\nMean IoU: {miou*100:.2f}%\nPixel Accuracy: {acc*100:.2f}%")
    return miou, acc

# ------------------------------
# Main
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    voc_root = r"C:\Users\sagef\OneDrive\Documents\Engineering\ELEC 475\LAB 3\data"
    val_dataset = datasets.VOCSegmentation(
        root=voc_root,
        year="2012",
        image_set="val",
        download=False,
        transform=transform,
        target_transform=lambda x: torch.from_numpy(np.array(x, dtype=np.int64))
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = models.segmentation.fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
    model.to(device)

    evaluate(model, val_loader, device, num_classes=21, visualize=True)

if __name__ == "__main__":
    main()
