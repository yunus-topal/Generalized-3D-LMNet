from torchvision.models import efficientnet_b0
import torch.nn as nn
import torch


class EfficientNetModified(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.model = efficientnet_b0(weights="DEFAULT")
        num_features = self.model.classifier[1].in_features  # Get the number of input features for the classifier
        self.model.classifier = nn.Sequential(
                                nn.Linear(num_features, latent_size),
                                )
        self.lin1 = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),

            nn.Linear(latent_size, latent_size),
            nn.ReLU(),     

            nn.Linear(latent_size, latent_size),
        )
        
        self.lin2 = nn.Sequential(
            nn.Linear(latent_size * 2, latent_size),
            nn.ReLU(),

            nn.Linear(latent_size, latent_size),
            nn.ReLU(),     

            nn.Linear(latent_size, latent_size),
        )

        self.lin3 = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),

            nn.Linear(latent_size, latent_size),
            nn.ReLU(),     

            nn.Linear(latent_size, latent_size),
        )


    def forward(self, x):
        x = self.model(x)
        return x
    
    def aggregate(self, tensor):
        mean = torch.mean(tensor, axis=1) # (N, T, 512) -> (N, 512) 
        global_info = self.lin1(mean) # (N, 512) -> (N, 512)

        # combine global info with tensor
        tensor = torch.cat((tensor, global_info.unsqueeze(1).repeat(1, tensor.shape[1], 1)), dim=2) # (N, T, 512) -> (N, T, 1024)
        tensor = self.lin2(tensor) # (N, T, 1024) -> (N, T, 512)
        tensor = torch.mean(tensor, axis=1) # (N, T, 512) -> (N, 512)
        tensor = self.lin3(tensor) # (N, 512) -> (N, 512)
        return tensor
        

def get_model(latent_size, aggregation_simple=True):
    if aggregation_simple == True:
        model = efficientnet_b0(weights="DEFAULT")

        num_features = model.classifier[1].in_features  # Get the number of input features for the classifier

        dropout_rate = 0.2
        model.classifier = nn.Sequential(
                                nn.Linear(num_features, latent_size),
                                )

        return model
    else:
        return EfficientNetModified(latent_size)

