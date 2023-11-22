import torch.nn as nn
import torch
import numpy as np
import config as c


class decodingNetwork(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(decodingNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.PixelUnshuffle(c.psf),
            nn.Conv2d(input_channel, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Sigmoid(),
            # nn.SiLU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(down_ratio_l2, down_ratio_l2), padding=(1, 1)),
            # nn.Sigmoid(),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.SiLU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96, output_channels, kernel_size=(3, 3), stride=(down_ratio_l3, down_ratio_l3), padding=(1, 1)),
            nn.Sigmoid(),
            nn.PixelShuffle(c.psf)
        )

    def forward(self, input):
        secret_rev = self.layers(input)
        return secret_rev
    

class dec_img(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(dec_img, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Sigmoid(),
            # nn.SiLU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(down_ratio_l2, down_ratio_l2), padding=(1, 1)),
            # nn.Sigmoid(),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.SiLU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96, output_channels, kernel_size=(3, 3), stride=(down_ratio_l3, down_ratio_l3), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, input):
        secret_rev = self.layers(input)
        return secret_rev




class dec_img_new(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(dec_img_new, self).__init__()

        self.layers = nn.Sequential(    
            nn.Conv2d(input_channel, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.SiLU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(down_ratio_l2, down_ratio_l2), padding=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.SiLU(inplace=True),
            nn.InstanceNorm2d(96),
            nn.Conv2d(96, output_channels, kernel_size=(3, 3), stride=(down_ratio_l3, down_ratio_l3), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, input):
        secret_rev = self.layers(input)
        return secret_rev



class stegan_dec(nn.Module):
    def __init__(self, input_channel=3, output_channels=3, down_ratio_l2=1, down_ratio_l3=1):
        super(stegan_dec, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, output_channels, kernel_size=(3, 3), stride=(down_ratio_l3, down_ratio_l3), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, input):
        secret_rev = self.layers(input)
        return secret_rev



def init_weights(model, random_seed=None):
    if random_seed != None:
        torch.manual_seed(random_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
            # nn.init.orthogonal_(m.weight, gain=1) 
            # nn.init.uniform_(m.weight, a=0, b=1) 

             
            # used in fnns for hiding binary bits stream
            # m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, m.weight.shape)).float())  

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight)
            nn.init.constant_(m.bias, 0)

