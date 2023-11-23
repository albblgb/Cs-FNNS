""" This module creates SRNet model."""
import torch
from torch import Tensor
from torch import nn
import sys
sys.path.append('./steganalysis_networks/srnet')
# import sys
# sys.path.append('./steganalysis_networks/siastegnet')
from model.utils import Type1, Type2, Type3, Type4


class Srnet(nn.Module):
    """This is SRNet model class."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.type1s = nn.Sequential(Type1(3, 64), Type1(64, 16))
        self.type2s = nn.Sequential(
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
        )
        self.type3s = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        self.type4 = Type4(256, 512)
        self.dense = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns logits for input images.
        Args:
            inp (Tensor): input image tensor of shape (Batch, 1, 256, 256)
        Returns:
            Tensor: Logits of shape (Batch, 2)
        """
        # print(inp.shape)
        out = self.type1s(inp)
        out = self.type2s(out)
        out = self.type3s(out)
        out = self.type4(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        # bk
        out = self.dense(out)
        # print('888')
        # print(out)
        # print('888')
        # return self.softmax(out)
        return out


if __name__ == "__main__":
    image = torch.randn((1, 1, 256, 256))
    net = Srnet()
    print(net(image).shape)
