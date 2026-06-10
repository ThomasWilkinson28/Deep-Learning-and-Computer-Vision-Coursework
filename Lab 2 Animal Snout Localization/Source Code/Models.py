import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import AlexNet_Weights, VGG16_Weights

# -----------------------------
# VGG16 Backbone for Regression
# -----------------------------
class SnoutNetVGG(nn.Module):
    def __init__(self, pretrained=True):
        super(SnoutNetVGG, self).__init__()
        # Load pretrained weights (PyTorch 2.6+) way
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg16 = models.vgg16(weights=weights)

        # Keep feature extractor (conv layers)
        self.features = vgg16.features  # Output shape: (batch, 512, 7, 7) for 224x224 input

        # Adapt classifier for regression
        # Original VGG16 classifier: [4096, 4096, 1000]
        self.regressor = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# -----------------------------
# AlexNet Backbone for Regression
# -----------------------------
class SnoutNetAlexNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SnoutNetAlexNet, self).__init__()
        # Load pretrained weights the new (PyTorch 2.6+) way
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        alexnet = models.alexnet(weights=weights)


        # Keep feature extractor (conv layers)
        self.features = alexnet.features  # Output shape: (batch, 256, 6, 6) for 224x224 input

        # Adapt classifier for regression
        # Original AlexNet classifier: [4096, 4096, 1000]
        self.regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 2)  # Final output: (x, y)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x



class SnoutNet(nn.Module):
    def __init__(self):
        super(SnoutNet, self).__init__()

        # Conv1: input (3,227,227) → output (64,227,227)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  
        #Added padding to maxpool so output would round to 57
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding = 1)  # -> (64, 57, 57)

        # Conv2: (64,57,57) → (128,57,57)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        #Added padding to maxpool so output would round to 15
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding = 2)  # -> (128, 15, 15)

        # Conv3: (128,15,15) → (256,15,15)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        #Added padding to maxpool so output would round to 4
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding = 1)  # -> (256, 4, 4)

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(4096, 1024) # -> (1024)
        self.fc2 = nn.Linear(1024, 1024) # -> (1024)
        self.fc3 = nn.Linear(1024, 2) # -> (2)

    def forward(self, x):
        # --- Convolutional Layers ---
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten (batch_size, 4096)
        x = x.view(x.size(0), -1)

        # --- Fully Connected Layers ---
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# --- Test Script ---
if __name__ == "__main__":
    # Create model
    model = SnoutNet()
    
    # Dummy input tensor (batch_size=1, RGB=3, width=227, height=227)
    dummy_input = torch.randn(1, 3, 227, 227)
    
    # Forward pass
    output = model(dummy_input)
    
    # Print the final output shape
    print("Output tensor shape:", output.shape)