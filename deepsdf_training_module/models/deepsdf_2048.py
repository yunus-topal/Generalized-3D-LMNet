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

class SmallDeepSDFDecoder(nn.Module):

    def __init__(self, size) -> None:
        super().__init__()

        dropout_prob = 0

        self.lin1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(size, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 1024-(size))),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
            
        )

        self.lin2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(1024, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 512)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(512, 256)),
            
        )

    def forward(self, x_in):
        x = self.lin1(x_in)
        x = self.lin2(torch.cat((x, x_in), dim=1))
        return x

class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        self.latent_size = latent_size
        dropout_prob = 0.0

        size = latent_size//4 + 3
        self.small_deep_sdf_decoder_1 = SmallDeepSDFDecoder(size)
        self.small_deep_sdf_decoder_2 = SmallDeepSDFDecoder(size)
        self.small_deep_sdf_decoder_3 = SmallDeepSDFDecoder(size)
        self.small_deep_sdf_decoder_4 = SmallDeepSDFDecoder(size)

        self.last_layer = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(256*4, 256*2)),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.utils.weight_norm(nn.Linear(256*2, 1))
        )
        

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        
        # split x_in into 4 parts where each part is of size latent_size//4
        x_in1 = x_in[:, :self.latent_size//4]
        x_in2 = x_in[:, self.latent_size//4:2*self.latent_size//4]
        x_in3 = x_in[:, 2*self.latent_size//4:3*self.latent_size//4]
        x_in4 = x_in[:, 3*self.latent_size//4:4*self.latent_size//4]

        # add last 3 from x_in to x_in1 x_in2 x_in3 x_in4
        x_in1 = torch.cat((x_in1, x_in[:, -3:]), dim=1)
        x_in2 = torch.cat((x_in2, x_in[:, -3:]), dim=1)
        x_in3 = torch.cat((x_in3, x_in[:, -3:]), dim=1)
        x_in4 = torch.cat((x_in4, x_in[:, -3:]), dim=1)

        x1 = self.small_deep_sdf_decoder_1(x_in1) # B x 128
        x2 = self.small_deep_sdf_decoder_2(x_in2) # B x 128
        x3 = self.small_deep_sdf_decoder_3(x_in3) # B x 128
        x4 = self.small_deep_sdf_decoder_4(x_in4) # B x 128

        # # concatenate x1 x2 x3 x4
        x = torch.cat((x1, x2, x3, x4), dim=1) # B x 512
        # x = x1
        x = self.last_layer(x) # B x 1

        return x

