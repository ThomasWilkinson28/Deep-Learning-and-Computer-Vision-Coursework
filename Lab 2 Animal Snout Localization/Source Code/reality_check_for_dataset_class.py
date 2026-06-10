import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DatasetSetup import Resize, RandomHorizontalFlip, RandomRotation, SnoutDataset, RandomTranslation, GaussianBlur, ColorJitter

def reality_check(train_loader):
    # ---- Train DataLoader Reality Check ----
    print("=== TRAIN DATASET ===")
    for idx, (image, uv) in enumerate(train_loader):
        # image shape: [1, 3, 227, 227], uv shape: [1, 2]
        img_tensor = image[0]  # remove batch dim
        uv_coords = uv[0]      # remove batch dim
        uv_coords_list = [round(x, 3) for x in uv_coords.tolist()]
        
        img_name = train_dataset.data[idx][0]  # image filename
        
        print(f"Train Image {idx}: {img_name}, UV: {uv_coords.tolist()}")
        
        # Visualize once every 50 images
        if idx % 50 == 0:
            img_np = img_tensor.permute(1,2,0).numpy()  # CxHxW -> HxWxC
            plt.figure(figsize=(4,4))
            plt.imshow(img_np)
            plt.scatter(uv_coords[0].item(), uv_coords[1].item(), c='r', s=50, marker='x')
            plt.title(f"{img_name} - UV: {uv_coords_list}")
            plt.axis('off')
            plt.show()

train_transform = [Resize((227,227)), 
            RandomHorizontalFlip(p=0.5), RandomRotation(degrees=15),
            RandomTranslation(max_translate=0.1), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            GaussianBlur(blur_prob=0.5, noise_std=0.02)]

# Create datasets
train_dataset = SnoutDataset(
    img_dir= "oxford-iiit-pet-noses/images-original/images",
    label_file= "oxford-iiit-pet-noses/train_noses.txt",
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

reality_check(train_loader)
