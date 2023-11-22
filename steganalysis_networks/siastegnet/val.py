

import os
import torch
import torch.nn as nn
import sys
sys.path.append('./steganalysis_networks/siastegnet')
import src.models
from src.data import build_val_loader
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
val_cover_dir = "/data/gbli/works/steganalysis/data/test/cover"
val_stego_dir = "/data/gbli/works/steganalysis/data/test/stego"
batch_size = 1
pre_model = "/data/gbli/works/steganalysis/SiaStegNet-master/checkpoint/odih1/model_best.pth.tar"
alpha = 0.1
margin = 1.00

val_loader = build_val_loader(
    val_cover_dir, val_stego_dir, batch_size=batch_size
)

net = src.models.KeNet().cuda()
net.load_state_dict(torch.load(pre_model, map_location='cuda')['state_dict'], strict=False)
                                                                                                                                                                                                                                
criterion_1 = nn.CrossEntropyLoss()
criterion_2 = src.models.ContrastiveLoss(margin=margin)

def preprocess_data(images, labels, random_crop):
    # images of shape: NxCxHxW
    if images.ndim == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
        labels = labels.squeeze(0)
    h, w = images.shape[-2:]

    if random_crop:
        ch = random.randint(h * 3 // 4, h)  # h // 2      #256
        cw = random.randint(w * 3 // 4, w)  # square ch   #256

        h0 = random.randint(0, h - ch)  # 128
        w0 = random.randint(0, w - cw)  # 128
    else:
        ch, cw, h0, w0 = h, w, 0, 0

    
    cw = cw & ~1
    inputs = [
        images[..., h0:h0 + ch, w0:w0 + cw // 2],
        images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
    ]
    
    inputs = [x.cuda() for x in inputs]
    labels = labels.cuda()
    return inputs, labels


def valid():
    net.eval()
    valid_loss = 0.
    valid_accuracy = 0.
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = preprocess_data(data['image'], data['label'],False)
            outputs, feats_0, feats_1 = net(*inputs)
            valid_loss += criterion_1(outputs, labels).item() + \
                            alpha * criterion_2(feats_0, feats_1, labels)
            
            valid_accuracy += src.models.accuracy(outputs, labels).item()
    valid_loss /= len(val_loader)
    valid_accuracy /= len(val_loader)
    return valid_loss, valid_accuracy

if __name__ == '__main__':
    _, accuracy = valid()
    print("accuracy: ", accuracy)


