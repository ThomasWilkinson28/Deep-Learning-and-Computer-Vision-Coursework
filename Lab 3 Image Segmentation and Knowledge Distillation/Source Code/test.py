"""
test.py
-------
Evaluate a trained LightweightSegModel on the Pascal VOC2012 dataset.
Displays mIoU, pixel accuracy, average inference speed, and example visualizations.

Usage:
    python test.py --weights checkpoints/best_model.pth --data ./VOCdevkit/VOC2012 --num-classes 21
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import time
from custom_model import LightweightUNetSeg


# ------------------------------
# Utility: color map for VOC
# ------------------------------
def voc_colormap(n=21):
    """Generate PASCAL VOC color map."""
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
# mIoU + Accuracy computation
# ------------------------------
def compute_metrics(preds, gts, num_classes=21):
    """Compute mean IoU and pixel accuracy."""
    ious, total_correct, total_label = [], 0, 0
    for pred, gt in zip(preds, gts):
        mask = gt != 255  # ignore label
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
    """Display input, prediction, and ground truth."""
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

    total_time = 0.0
    num_images = 0

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)

            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()

            logits = model(imgs)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()

            total_time += (end_time - start_time)
            num_images += imgs.size(0)

            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            preds_all.extend(preds.cpu().numpy())
            gts_all.extend(targets.cpu().numpy())

            if visualize and i < 1:  # visualize first few samples
                visualize_sample(imgs[0].cpu(), preds[0].cpu().numpy(), targets[0].cpu().numpy(), cmap)

    avg_time = total_time / num_images
    avg_ms = avg_time * 1000
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    miou, acc = compute_metrics(preds_all, gts_all, num_classes)
    print(f"\n✅ Evaluation Results:\n"
          f"Mean IoU: {miou * 100:.2f}%\n"
          f"Pixel Accuracy: {acc * 100:.2f}%\n"
          f"Average Inference Time: {avg_ms:.2f} ms per image\n"
          f"Throughput: {fps:.2f} FPS")
    return miou, acc, avg_ms, fps


# ------------------------------
# Main
# ------------------------------
def main():
    weights = r"C:\Users\sagef\OneDrive\Documents\Engineering\ELEC 475\LAB 3\submission\weights_custom_model.pth"
    data = r"C:\Users\sagef\OneDrive\Documents\Engineering\ELEC 475\LAB 3\data"
    num_classes = 21
    batch_size = 1  # fix for variable image sizes
    visualize = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LightweightUNetSeg(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)

    # Dataset
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_dataset = datasets.VOCSegmentation(
        root=data,
        year="2012",
        image_set="val",
        download=False,
        transform=val_transform,
        target_transform=lambda x: torch.from_numpy(np.array(x, dtype=np.int64))
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    evaluate(model, val_loader, device, num_classes=num_classes, visualize=visualize)


if __name__ == "__main__":
    main()
