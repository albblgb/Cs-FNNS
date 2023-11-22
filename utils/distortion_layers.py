import torch
import config as c
import torchvision
from io import BytesIO
from PIL import Image


def gaussian_noise_layer(adv_pert):
    return  adv_pert + torch.randn(adv_pert.shape).mul_(c.sigma/255).to(adv_pert.device)

def poisson_noise_layer(adv_pert):
    return  adv_pert + torch.poisson(torch.rand(adv_pert.shape)).mul_(c.sigma/255).to(adv_pert.device)


transform_to_pil = torchvision.transforms.ToPILImage()
transform_to_tensor = torchvision.transforms.ToTensor()
ps = torch.nn.PixelShuffle(c.psf)
pus = torch.nn.PixelUnshuffle(c.psf)

def jpeg_compression_layer(adv_pert, cover):

    adv_image = cover + adv_pert
    # adv_image = ps(adv_image)
    adv_image = adv_image.squeeze(dim=0).cpu()
    adv_image = transform_to_pil(adv_image)
    outputIoStream = BytesIO()
    adv_image.save(outputIoStream, "JPEG", quality=c.qf)
    outputIoStream.seek(0)
    adv_image_jpeg = Image.open(outputIoStream)
    adv_image_jpeg = transform_to_tensor(adv_image_jpeg).unsqueeze(dim=0).to(adv_pert.device)
    # adv_image_jpeg = pus(adv_image_jpeg)
    jpeg_noise = (adv_image_jpeg - (cover + adv_pert)).detach()
    
    return adv_pert + jpeg_noise


def attack_layer(adv_pert, cover):
    if c.attack_layer == 'gaussian':
        return gaussian_noise_layer(adv_pert)
    elif c.attack_layer == 'possion':
        return poisson_noise_layer(adv_pert)
    else: # jpeg
        return jpeg_compression_layer(adv_pert, cover)



def img_jpeg_compression(adv_image):
    # adv_image: tensor
    device = adv_image.device
    # adv_image = transform_to_pil(ps(adv_image).squeeze(dim=0).cpu())
    adv_image = transform_to_pil(adv_image.squeeze(dim=0).cpu())
    outputIoStream = BytesIO()
    adv_image.save(outputIoStream, "JPEG", quality=c.qf)
    outputIoStream.seek(0)
    adv_image_jpeg = Image.open(outputIoStream)
    adv_image_jpeg = transform_to_tensor(adv_image_jpeg).unsqueeze(dim=0).to(device)

    return adv_image_jpeg




# '''
#     'gauss'     Gaussian-distributed additive noise.
#     'poisson'   Poisson-distributed noise generated from the data.
#     's&p'       Replaces random pixels with 0 or 1.
# '''




