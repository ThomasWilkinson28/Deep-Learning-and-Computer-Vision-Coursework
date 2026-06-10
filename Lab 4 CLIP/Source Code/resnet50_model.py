import torch.nn as nn
import torchvision.models as models
import torch

class ImageEncoder(nn.Module):
    """
    A ResNet50-based image encoder with a projection head to 
    map image features into the CLIP embedding space.
    
    Args:
        embedding_dim (int): The target dimension for the output embeddings 
                             (default: 512, to match CLIP).
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # Load a pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Get the number of input features for the final linear layer
        in_features = self.model.fc.in_features # This is 2048 for ResNet50
        
        # Replace the final classification layer with an Identity layer
        # We will add our own projection head
        self.model.fc = nn.Identity()
        
        # Define the 2-layer projection head
        self.projection = nn.Sequential(
            # 1st linear layer
            nn.Linear(in_features, in_features), # [2048 -> 2048]
            nn.GELU(),                          # GELU activation
            # 2nd linear layer
            nn.Linear(in_features, embedding_dim) # [2048 -> 512]
        )

    def forward(self, x):
        """
        Forward pass:
        1. Get ResNet features (before original classification head)
        2. Pass features through the new projection head
        """
        features = self.model(x)
        embedding = self.projection(features)
        return embedding

class ImageEncoderModified(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.5):
        super().__init__()
        
        # Load ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Feature dimension (2048)
        in_features = self.model.fc.in_features 
        self.model.fc = nn.Identity()
        
        # IMPROVED PROJECTION HEAD
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features), # Added Batch Norm
            nn.GELU(),
            nn.Dropout(p=dropout),       # Added Dropout
            nn.Linear(in_features, embedding_dim)
        )

    def forward(self, x):
        features = self.model(x)
        embedding = self.projection(features)
        return embedding


if __name__ == '__main__':
    # Simple test to check model dimensions
    model_to_test = 'normal'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_to_test == 'improved':
        print("Testing ImageEncoderImproved...")
        model = ImageEncoderImproved(embedding_dim=512).to(device)
    else:
        print("Testing ImageEncoder...")
        model = ImageEncoder(embedding_dim=512).to(device)
    # Create a dummy image batch [Batch, Channels, Height, Width]
    dummy_image = torch.randn(4, 3, 224, 224).to(device)
    
    # Pass through the model
    output_embedding = model(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output embedding shape: {output_embedding.shape}")
    
    assert output_embedding.shape == (4, 512)
    print("Test passed! Model output dimensions are correct.")

