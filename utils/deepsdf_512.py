import torch.nn as nn
import torch

class LinearResnetBlock(nn.Module):

    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch

        self.lin1 = nn.Linear(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.lin3 = nn.Linear(mid_ch, in_ch)
        self.relu3 = nn.ReLU(inplace=True) 

    def forward(self, x):
        x_in = x
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        x = self.relu3(x)
        x = x + x_in
        return x


class DeepSDFDecoder512(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.0

        # TODO: Define model
        self.lin1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(latent_size + 3, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 1024-(latent_size + 3))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        # concatenate
        self.lin2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 128)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(128, 1)
        )
        

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        x1 = self.lin1(x_in)
        x2 = self.lin2(torch.cat((x1, x_in), dim=1))
        return x2

class DeepSDFDecoder512v2(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.0

        # TODO: Define model
        self.lin1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(latent_size + 3, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024-(latent_size + 3))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        
        self.lin2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024-(latent_size + 3))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        self.lin3 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024-(latent_size + 3))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        # concatenate
        self.lin_final = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(1024, 1)
        )
        

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        x1 = self.lin1(x_in)
        x2 = self.lin2(torch.cat((x1, x_in), dim=1))
        x3 = self.lin3(torch.cat((x2, x_in), dim=1))
        x4 = self.lin_final(torch.cat((x3, x_in), dim=1))
        return x4

class DeepSDFDecoder512v2_dropout(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.1

        # TODO: Define model
        self.lin1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(latent_size + 3, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024-(latent_size + 3))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        
        self.lin2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024-(latent_size + 3))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        self.lin3 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024-(latent_size + 3))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        # concatenate
        self.lin_final = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(1024, 1)
        )
        

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        x1 = self.lin1(x_in)
        x2 = self.lin2(torch.cat((x1, x_in), dim=1))
        x3 = self.lin3(torch.cat((x2, x_in), dim=1))
        x4 = self.lin_final(torch.cat((x3, x_in), dim=1))
        return x4
