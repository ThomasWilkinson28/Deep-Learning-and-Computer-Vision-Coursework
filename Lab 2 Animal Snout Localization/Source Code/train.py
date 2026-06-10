import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from torch.utils.data import DataLoader
from Models import SnoutNet, SnoutNetVGG, SnoutNetAlexNet
from DatasetSetup import SnoutDataset, Resize, RandomHorizontalFlip, RandomRotation, RandomTranslation, ColorJitter, GaussianBlur

def train(n_epochs, optimizer, model, loss_fn, train_loader, validation_loader, device, save_file, plot_file):
    print('training ...')
    start_time = datetime.now()
    print("Start Time:", start_time.strftime("%H:%M:%S"))
    train_losses = []
    val_losses = []
    
    # initialize figure once for efficiency
    plt.figure(figsize=(10, 6))

    for epoch in range(1, n_epochs+1):
        print('Epoch ', epoch)
        model.train() 
        running_loss = 0.0
        
        for imgs, coordinates in train_loader:
            scaled_coordinates = coordinates / 227.0 
            imgs, scaled_coordinates = imgs.to(device), scaled_coordinates.to(device)
            
            optimizer.zero_grad() # reset optimizer gradients to zero
            
            outputs = model(imgs) # forward propagation through network
            loss = loss_fn(outputs, scaled_coordinates) # calculate loss
            loss.backward() # calculate loss gradients
            optimizer.step() # iterate the optimization, based on loss gradients
            
            running_loss += loss.item() # update value losses
        
        avg_loss_train = running_loss/len(train_loader.dataset)
        train_losses.append(avg_loss_train) # update value of losses
        
        # --- Validation ---
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for imgs, coordinates in validation_loader:
                scaled_coordinates = coordinates / 227.0  
                imgs, scaled_coordinates = imgs.to(device), scaled_coordinates.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, scaled_coordinates)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(validation_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"| Epoch {epoch} | "
              f"Train Loss: {avg_loss_train:.6f} | Val Loss: {avg_val_loss:.6f}")
        if save_file:
                torch.save(model.state_dict(), save_file)
                
        if plot_file:
            plt.clf()
            plt.plot(train_losses, label="Train Loss", color="steelblue")
            plt.plot(val_losses, label="Val Loss", color="orange")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc="upper right")
            plt.title("Training & Validation Loss")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_file)
    
    end_time = datetime.now()
    print("End:", end_time.strftime("%H:%M:%S"))

    # Compute elapsed time
    elapsed = end_time - start_time

    # Convert to hours, minutes, seconds
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Total run time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # He (Kaiming) initialization for ReLU layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.Linear):
        # If it's the final output layer (2 outputs for x,y)
        if m.out_features == 2:
            # Xavier with small gain to start predictions near zero
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)
        else:
            # He (Kaiming) initialization for ReLU hidden layers
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def init_regressor_weights(m):
    if isinstance(m, nn.Linear):
        # If it's the final output layer (2 outputs for x,y)
        if m.out_features == 2:
            # Xavier with small gain to start predictions near zero
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)
        else:
            # He (Kaiming) initialization for ReLU hidden layers
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        

def main():
    #Default settings which can be overwritten by command line
    n_epochs = 50
    batch_size = 16
    augmentation= 'unaugmented'
    model_name = 'SnoutNet'
    plot_file = 'plot.png'
    lr = 1e-3 

    print('running main ...')

    #Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch', type=int, help='batch size')
    argParser.add_argument('-a', metavar='state', type=str, help='with or without augmentation')
    argParser.add_argument('-m', metavar='state', type=str, help='model')
   
    args = argParser.parse_args()

    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.a != None:
        augmentation = args.a
    if args.m != None:
        model_name = args.m
    
    save_file = f"{model_name}_{augmentation}_weights.pth"
    plot_file = f"{model_name}_{augmentation}_plot.png"

    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)
    print('\t\tmodel = ', model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'SnoutNet':
        model = SnoutNet().to(device)
        model.apply(init_weights)
    elif model_name == 'SnoutNet-A':
        model = SnoutNetAlexNet().to(device)
        model.apply(init_regressor_weights)
    elif model_name == 'SnoutNet-V':
        model = SnoutNetVGG().to(device)
        model.apply(init_regressor_weights) 
    
    #For training
    if augmentation == 'augmented':
        print('Training With Augmentation')
        train_transform = [Resize((227,227)), 
            RandomHorizontalFlip(p=0.5), RandomRotation(degrees=15),
            RandomTranslation(max_translate=0.1), 
            #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            #GaussianBlur(blur_prob=0.5, noise_std=0.02)
            ]
    else:
        print('No Augmentation')
        train_transform = [Resize((227, 227))]

    val_transform = [Resize((227,227))]
    
    # Create datasets
    train_dataset = SnoutDataset(
    img_dir= "oxford-iiit-pet-noses/images-original/images",
    label_file= "oxford-iiit-pet-noses/train_noses.txt",
    transform=train_transform
    )

    val_dataset = SnoutDataset(
        img_dir="oxford-iiit-pet-noses/images-original/images/",
        label_file="oxford-iiit-pet-noses/test_noses.txt",
        transform=val_transform
    )
    
    validation_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            save_file=save_file,
            plot_file = plot_file)

###################################################################

if __name__ == '__main__':
    main()



