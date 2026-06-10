import torch
from torch.utils.data import DataLoader
from DatasetSetup import SnoutDataset
from Models import SnoutNet,SnoutNetAlexNet, SnoutNetVGG
import matplotlib.pyplot as plt
import numpy as np
import random
from DatasetSetup import SnoutDataset, Resize
import argparse
from datetime import datetime

def visualize_ensemble_predictions(test_dataset, device,
                                   unaug_paths, aug_paths,
                                   unaug_weights, aug_weights,
                                   num_samples=5):
    """
    Visualizes side-by-side predictions from the unaugmented and augmented ensembles.
    Each row = one sample, left = unaugmented ensemble, right = augmented ensemble.
    """
    
    # Initialize the three models
    models_list = [
        SnoutNet().to(device),
        SnoutNetAlexNet().to(device),
        SnoutNetVGG().to(device)
    ]

    # Pick random samples from dataset
    indices = random.sample(range(len(test_dataset)), num_samples)
    plt.figure(figsize=(10, num_samples * 3))

    # Helper to compute weighted ensemble prediction for one sample
    def get_ensemble_prediction(img_input, weight_paths, weights):
        preds = []
        for i, model in enumerate(models_list):
            model.load_state_dict(torch.load(weight_paths[i], map_location=device))
            model.eval()
            with torch.no_grad():
                pred = model(img_input) * weights[i]
            preds.append(pred)
        combined_pred = torch.stack(preds).sum(dim=0)
        return (combined_pred[0].cpu().numpy() * 227.0)  # scale to pixel coordinates

    for row_idx, idx in enumerate(indices):
        image, gt_uv = test_dataset[idx]
        img_input = image.unsqueeze(0).to(device)
        gt = gt_uv.cpu().numpy()

        # -------------------
        # Ensemble predictions
        # -------------------
        pred_unaug = get_ensemble_prediction(img_input, unaug_paths, unaug_weights)
        pred_aug = get_ensemble_prediction(img_input, aug_paths, aug_weights)

        # Normalize image for plotting
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # -------------------
        # Plot Left: Unaugmented Ensemble
        # -------------------
        plt.subplot(num_samples, 2, 2 * row_idx + 1)
        plt.imshow(img_np)
        plt.scatter(gt[0], gt[1], c='lime', s=40, label='GT')
        plt.scatter(pred_unaug[0], pred_unaug[1], c='red', s=40, marker='x', label='Pred')
        plt.title(f"Unaugmented Ensemble — Sample {row_idx + 1}")
        plt.axis('off')
        if row_idx == 0:
            plt.legend(loc='upper right')

        # -------------------
        # Plot Right: Augmented Ensemble
        # -------------------
        plt.subplot(num_samples, 2, 2 * row_idx + 2)
        plt.imshow(img_np)
        plt.scatter(gt[0], gt[1], c='lime', s=40, label='GT')
        plt.scatter(pred_aug[0], pred_aug[1], c='red', s=40, marker='x', label='Pred')
        plt.title(f"Augmented Ensemble — Sample {row_idx + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def ensemble(test_loader, device, unaug_paths, aug_paths,unaug_weights,aug_weights, visualize, start_time):
    """
    Combines predictions from SnoutNet, SnoutNetAlexNet, and SnoutNetVGG
    using weighted averaging to produce an ensemble prediction.
    """
    if visualize == 'Visualize':
        # assuming test_loader.dataset is the same dataset used for testing
        test_dataset = test_loader.dataset
        visualize_ensemble_predictions(test_dataset,device,unaug_paths,aug_paths,unaug_weights,aug_weights,num_samples=5)
    else:    
        # -------------------------------
        # Initialize models
        # -------------------------------
        models_list = [
            SnoutNet().to(device),
            SnoutNetAlexNet().to(device),
            SnoutNetVGG().to(device)
        ]

        # Function to evaluate a set of 3 models and return combined predictions
        def evaluate_model_group(weight_paths, weights):
            print(f"\nEvaluating Ensemble with weights: {weights}")
            for i, model in enumerate(models_list):
                model.load_state_dict(torch.load(weight_paths[i], map_location=device))
                model.eval()

            all_distances = []

            with torch.no_grad():
                for imgs, coordinates in test_loader:
                    imgs, coordinates = imgs.to(device), coordinates.to(device)

                    # Collect predictions from all models
                    preds = []
                    for i, model in enumerate(models_list):
                        pred = model(imgs)
                        preds.append(pred * weights[i])

                    # Weighted sum of predictions
                    combined_pred = torch.stack(preds).sum(dim=0)

                    # Scale predictions to pixel coordinates
                    preds_pixel = combined_pred * 227.0

                    # Euclidean distance between predicted and GT
                    distances = torch.sqrt(torch.sum((preds_pixel - coordinates) ** 2, dim=1))
                    all_distances.extend(distances.cpu().numpy())

            # Compute stats
            # Compute stats
            all_distances = np.array(all_distances)
            sorted_distances = np.sort(all_distances)

            # 4 best (lowest errors)
            best_4 = sorted_distances[:4]
            # 4 worst (highest errors)
            worst_4 = sorted_distances[-4:]

            return {
                'min': all_distances.min(),
                'mean': all_distances.mean(),
                'max': all_distances.max(),
                'std': all_distances.std(),
                'best_4': best_4,
                'worst_4': worst_4
            }
        

        # -------------------------------
        # Evaluate unaugmented ensemble
        # -------------------------------
        print("Running Unaugmented Ensemble Evaluation...")
        stats_unaug = evaluate_model_group(unaug_paths, unaug_weights)
        print(f"\nLocalization Error Statistics (Unaugmented Ensemble):")
        print(f"Min: {stats_unaug['min']:.2f}")
        print(f"Mean: {stats_unaug['mean']:.2f}")
        print(f"Max: {stats_unaug['max']:.2f}")
        print(f"Std: {stats_unaug['std']:.2f}")

        # Best / Worst Subsets
        print("\nTop 4 Best Predictions (Lowest Errors):")
        print(f"Values: {stats_unaug['best_4']}")
        print(f"Mean: {stats_unaug['best_4'].mean():.2f}, Std: {stats_unaug['best_4'].std():.2f}")

        print("\nTop 4 Worst Predictions (Highest Errors):")
        print(f"Values: {stats_unaug['worst_4']}")
        print(f"Mean: {stats_unaug['worst_4'].mean():.2f}, Std: {stats_unaug['worst_4'].std():.2f}")
        print("-----------------------------------------------------\n")
        # -------------------------------
        # Evaluate augmented ensemble
        # -------------------------------
        print("Running Augmented Ensemble Evaluation...")
        stats_aug = evaluate_model_group(aug_paths, aug_weights)
        print(f"\nLocalization Error Statistics (Augmented Ensemble):")
        print(f"Min: {stats_aug['min']:.2f}")
        print(f"Mean: {stats_aug['mean']:.2f}")
        print(f"Max: {stats_aug['max']:.2f}")
        print(f"Std: {stats_aug['std']:.2f}")

        # Best / Worst Subsets
        print("\nTop 4 Best Predictions (Lowest Errors):")
        print(f"Values: {stats_aug['best_4']}")
        print(f"Mean: {stats_aug['best_4'].mean():.2f}, Std: {stats_aug['best_4'].std():.2f}")

        print("\nTop 4 Worst Predictions (Highest Errors):")
        print(f"Values: {stats_aug['worst_4']}")
        print(f"Mean: {stats_aug['worst_4'].mean():.2f}, Std: {stats_aug['worst_4'].std():.2f}")

        end_time = datetime.now()
        print("End:", end_time.strftime("%H:%M:%S"))

        # Compute elapsed time
        elapsed = end_time - start_time

        # Convert to hours, minutes, seconds
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Total run time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")




def visualize_predictions_compared(model, test_dataset, device, weights_paths, num_samples=5):
    assert len(weights_paths) == 2, "Expected exactly two weight paths: [unaugmented, augmented]"

    unaug_path, aug_path = weights_paths

    # Randomly pick samples (same for both models)
    indices = random.sample(range(len(test_dataset)), num_samples)

    plt.figure(figsize=(10, num_samples * 3))

    # Prepare subplot grid: num_samples rows × 2 columns
    for row_idx, idx in enumerate(indices):
        image, gt_uv = test_dataset[idx]
        img_input = image.unsqueeze(0).to(device)
        gt_uv = gt_uv.to(device)
        gt = gt_uv.cpu().numpy()

        # -------------------
        # Unaugmented model
        # -------------------
        model.load_state_dict(torch.load(unaug_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred_unaug = model(img_input)[0].cpu().numpy() * 227.0

        # -------------------
        # Augmented model
        # -------------------
        model.load_state_dict(torch.load(aug_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred_aug = model(img_input)[0].cpu().numpy() * 227.0

        # Convert image for plotting
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # -------------------
        # Plot Left: Unaugmented
        # -------------------
        plt.subplot(num_samples, 2, 2 * row_idx + 1)
        plt.imshow(img_np)
        plt.scatter(gt[0], gt[1], c='lime', s=40, label='GT')
        plt.scatter(pred_unaug[0], pred_unaug[1], c='red', s=40, marker='x', label='Pred')
        plt.title(f"Unaugmented — Sample {row_idx + 1}")
        plt.axis('off')
        if row_idx == 0:
            plt.legend(loc='upper right')

        # -------------------
        # Plot Right: Augmented
        # -------------------
        plt.subplot(num_samples, 2, 2 * row_idx + 2)
        plt.imshow(img_np)
        plt.scatter(gt[0], gt[1], c='lime', s=40, label='GT')
        plt.scatter(pred_aug[0], pred_aug[1], c='red', s=40, marker='x', label='Pred')
        plt.title(f"Augmented — Sample {row_idx + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    #Default settings which can be overwritten by command line
    model_name = 'SnoutNet'
    augmentation = 'unaugmented'
    batch_size = 16
    visualize = 'False'
    weights_paths = []

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='state', type=str, help='model')
    argParser.add_argument('-b', metavar='batch', type=int, help='batch size')
    argParser.add_argument('-v', metavar='state', type=str, help='visualize bool')
    args = argParser.parse_args()

    if args.m != None:
        model_name = args.m
    if args.b != None:
        batch_size = args.b
    if args.v != None:
        visualize = args.v
    
    if model_name != 'SnoutNet-Ensemble':
        weights_paths = [f"Weights/{model_name}_unaugmented_weights.pth", f"Weights/{model_name}_augmented_weights.pth"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------
    # Load test dataset
    # -----------------------
    test_transform = [Resize((227, 227))]
    test_dataset = SnoutDataset(
        img_dir= "oxford-iiit-pet-noses/images-original/images",
        label_file= "oxford-iiit-pet-noses/test_noses.txt",
        transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------
    # Load trained model
    # -----------------------
    print('testing ...')
    start_time = datetime.now()
    print("Start Time:", start_time.strftime("%H:%M:%S"))
    if model_name == 'SnoutNet':
        model = SnoutNet().to(device)
    elif model_name == 'SnoutNet-A':
        model = SnoutNetAlexNet().to(device)
    elif model_name == 'SnoutNet-V':
        model = SnoutNetVGG().to(device)
    elif model_name == 'SnoutNet-Ensemble':
        unaug_paths = [
        "Weights/SnoutNet_unaugmented_weights.pth",
        "Weights/SnoutNet-A_unaugmented_weights.pth",
        "Weights/SnoutNet-V_unaugmented_weights.pth"]
        aug_paths = [
            "Weights/SnoutNet_augmented_weights.pth",
            "Weights/SnoutNet-A_augmented_weights.pth",
            "Weights/SnoutNet-V_augmented_weights.pth"]

        #Adjust weights based on known performance (example: VGG performs best)
        unaug_weights = (0.01, 0.05, 0.94)
        aug_weights = (0.01, 0.04, 0.95)
        ensemble(test_loader, device, unaug_paths, aug_paths, unaug_weights, aug_weights, visualize, start_time)
    
    #If testing for any model other than Ensemble weights_path will return true
    if weights_paths:
        #If user has selected to visualize the following if statement will return true
        if visualize == 'Visualize':
                visualize_predictions_compared(model, test_dataset, device, weights_paths, num_samples=5) 
        else:
            for weights_path in weights_paths:    
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.eval()

                # -----------------------
                # Testing loop
                # -----------------------
                all_distances = []

                with torch.no_grad():
                    for imgs, coordinates in test_loader:
                        imgs, coordinates = imgs.to(device), coordinates.to(device)
                        # scale coordinates if trained on normalized targets
                        preds = model(imgs)

                        # scale predictions back to pixel coords
                        preds_pixel = preds * 227.0

                        # compute Euclidean distance per sample
                        distances = torch.sqrt(torch.sum((preds_pixel - coordinates) ** 2, dim=1))
                        all_distances.extend(distances.cpu().numpy())

                
                weights_split = weights_path.split('_')
                augmentation = weights_split[1]
                
                # -----------------------
                # Statistics
                # -----------------------
                all_distances = np.array(all_distances)
                
                # Best/Worst prediction subsets
                sorted_distances = np.sort(all_distances)
                best_4 = sorted_distances[:4]
                worst_4 = sorted_distances[-4:]
                
                print(f"Error Stats (in pixels) for the {augmentation} {model_name} model:")
                print(f"Min: {all_distances.min():.2f}")
                print(f"Mean: {all_distances.mean():.2f}")
                print(f"Max: {all_distances.max():.2f}")
                print(f"Std: {all_distances.std():.2f}")

                print("Top 4 Best Predictions (Lowest Errors):")
                print(f"Values: {best_4}")
                print(f"Mean: {best_4.mean():.2f}, Std: {best_4.std():.2f}")

                print("\nTop 4 Worst Predictions (Highest Errors):")
                print(f"Values: {worst_4}")
                print(f"Mean: {worst_4.mean():.2f}, Std: {worst_4.std():.2f}")
                print("-----------------------------------------------------\n")
            
            end_time = datetime.now()
            print("End:", end_time.strftime("%H:%M:%S"))

            # Compute elapsed time
            elapsed = end_time - start_time

            # Convert to hours, minutes, seconds
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"Total run time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")



###################################################################

if __name__ == '__main__':
    main()

