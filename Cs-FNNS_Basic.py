import numpy as np 
import torch
from imageio.v2 import imread
from torch import nn
import random
import argparse
import os
from PIL import Image
import torch.nn.functional as F
from math import log10
from skimage.transform import resize
import torchvision
import glob
import cv2
from natsort import natsorted

from models.decodingNetwork import decodingNetwork
from models.network_dncnn import DnCNN
from utils.model import init_weights
from utils.image import calculate_ssim, calculate_psnr, calculate_mae
from utils.logger import logging, logger_info
from utils.dir import mkdirs
from utils.draw import img_hist
import config as c


secret_dataset = c.secret_dataset_dir.split('/')[-2]
cover_dataset = c.cover_dataset_dir.split('/')[-2]

logger_name = 'Cs-FNNS_basic'
image_save_dirs = os.path.join('results', 'Cs-FNNS_basic', secret_dataset)
mkdirs(image_save_dirs)
logger_info(logger_name, log_path=os.path.join(image_save_dirs, 'result' + '.log'))
logger = logging.getLogger(logger_name)
logger.info('secret dataset: {:s}'.format(secret_dataset))
logger.info('cover dataset: {:s}'.format(cover_dataset))
logger.info('beta: {:.2f}'.format(c.beta))
logger.info('learning rate: {:.3f}'.format(c.lr))
logger.info('epsilon: {:.2f}'.format(c.eps))
logger.info('number of iterations: {}'.format(c.iters))
logger.info('the size of secret image: {}'.format(c.secret_image_size))
logger.info('the size of cover image: {}'.format(c.cover_image_size))
os.environ["CUDA_VISIBLE_DEVICES"] = c.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


l_rev = torch.nn.MSELoss()
l_hid = torch.nn.MSELoss()


if c.cover_image_size // c.secret_image_size == 1:
    down_ratio_l3 = 1; down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 2:
    down_ratio_l3 = 2; down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 4:
    down_ratio_l3 = 2; down_ratio_l2 = 2
else:
    print('The code does not take into account the current situation, please adjust the image resulation')


# parparing decoder and denosing model
import PerceptualSimilarity.models
model = decodingNetwork(input_channel=3*c.psf*c.psf, output_channels=3*c.psf*c.psf, down_ratio_l2=down_ratio_l2, down_ratio_l3=down_ratio_l3).to(device)
denoise_model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R').to(device)
denoise_model.load_state_dict(torch.load('models/dncnn_color_blind.pth'), strict=True)
LpipsNet = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0]) # For calculating LPIPS


secret_image_path_list = list(natsorted(glob.glob(os.path.join(c.secret_dataset_dir, '*'))))
cover_image_path_list = list(natsorted(glob.glob(os.path.join(c.cover_dataset_dir, '*'))))


stego_psnr_list = []; stego_ssim_list = []; stego_lpips_list = []; stego_apd_list = []
secret_rev_psnr_list = []; secret_rev_ssim_list = []; secret_rev_lpips_list = []; secret_rev_apd_list = []
for i in range(len(secret_image_path_list)):  

    logger.info('*'*60)
    logger.info('hiding {}-th image'.format(i))
    
    # load secret image
    secret = imread(secret_image_path_list[i], pilmode='RGB') / 255.0 
    secret = resize(secret, (c.secret_image_size, c.secret_image_size))
    secret = torch.FloatTensor(secret).permute(2, 1, 0).unsqueeze(0).to(device)

    # load cover image
    cover = imread(cover_image_path_list[i], pilmode='RGB') / 255.0
    cover = resize(cover, (c.cover_image_size, c.cover_image_size))
    cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0).to(device)

    # praparing decoder
    random_seed_for_decodor = random.randint(0, 100000000)
    logger.info('random_seed_for_decodor(receiver): {:s}'.format(str(random_seed_for_decodor)))
    init_weights(model, random_seed_for_decodor)
    model = model.to(device)
    model.eval()

    # init perturbation
    w_pert=torch.autograd.Variable(torch.zeros_like(cover).float()).to(device)
    w_pert.requires_grad = True
    w_zero=torch.autograd.Variable(torch.zeros_like(cover).float()).to(device)

    # get the lower and upper bound of the perturbation
    mask_pos = torch.gt((torch.ones_like(cover) - cover), (torch.ones_like(cover) * c.eps)).int()
    mask_neg = torch.gt(cover, (torch.ones_like(cover) * c.eps)).int()
    U = (torch.ones_like(cover) * c.eps) * mask_pos + (torch.ones_like(cover) - cover) * (1 - mask_pos)
    L = -1 * ((torch.ones_like(cover) * c.eps) * mask_neg + cover * (1 - mask_neg))

    # optimizator
    optimizer = torch.optim.Adam([w_pert], lr=c.lr)   
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.5)
    
    for iteration_index in range(c.iters):
        optimizer.zero_grad()
        
        adv_pert = L + (U-L)*((torch.tanh(w_pert) + 1)/2)

        output = model(adv_pert)     
        loss_1 = l_hid(adv_pert, w_zero)  
        loss_2 = l_rev(output, secret)            
        loss =  loss_1 + c.beta * loss_2                        
        loss.backward(retain_graph=True)
        optimizer.step()
        weight_scheduler.step()

    logger.info('-'*60)
    adv_pert = L + (U-L)*((torch.tanh(w_pert) + 1)/2)

    # rounding and clipping operations
    adv_image = cover + adv_pert
    adv_image = torch.round(torch.clamp(adv_image*255, min=0., max=255.))/255
    adv_pert = adv_image - cover
    # testing 
    secret_rev = model(adv_pert)


    # denosing the decoded secret images to obtain recoverd images
    secret_rev = denoise_model(secret_rev)
    secret_rev = torch.round(torch.clamp(secret_rev*255, min=0., max=255.))/255
    cover_resi = (adv_image - cover).abs() * c.resi_magnification
    secret_resi = (secret_rev - secret).abs() * c.resi_magnification


    # calculing lpips
    stego_lpips = LpipsNet.forward(cover, adv_image, normalize=True)
    secret_rev_lpips = LpipsNet.forward(secret, secret_rev, normalize=True)


    # tensor(cuda) to numpy(cpu)
    secret_rev_512 = resize(secret_rev.clone().squeeze().permute(2,1,0).detach().cpu().numpy()*255, (512, 512))
    secret_rev_512 = torch.FloatTensor(secret_rev_512/255).permute(2,1,0).unsqueeze(dim=0).to(device)
    secret_512 = imread(secret_image_path_list[i], pilmode='RGB') / 255.0 
    secret_512 = resize(secret_512, (c.cover_image_size, c.cover_image_size))
    secret_512 = torch.FloatTensor(secret_512).permute(2, 1, 0).unsqueeze(0).to(device)
    secret_rev_512_lpips = LpipsNet.forward(secret_512, secret_rev_512, normalize=True)

    cover = cover.clone().squeeze().permute(2,1,0).detach().cpu().numpy() * 255
    stego = adv_image.clone().squeeze().permute(2,1,0).detach().cpu().numpy() * 255
    secret = secret.clone().squeeze().permute(2,1,0).detach().cpu().numpy() * 255
    secret_rev = secret_rev.clone().squeeze().permute(2,1,0).detach().cpu().numpy() * 255
    cover_resi = cover_resi.clone().squeeze().permute(2,1,0).detach().cpu().numpy() * 255
    secret_resi = secret_resi.clone().squeeze().permute(2,1,0).detach().cpu().numpy() * 255


    # calculing and recoding metrics
    stego_psnr = calculate_psnr(cover, stego)
    stego_ssim = calculate_ssim(cover, stego)
    secret_rev_psnr = calculate_psnr(secret, secret_rev)
    secret_rev_ssim = calculate_ssim(secret, secret_rev)
    stego_apd = calculate_mae(cover, stego)
    secret_rev_apd = calculate_mae(secret, secret_rev)
    logger.info('stego_psnr: {:.2f}, secret_rev_psnr: {:.2f}'.format(stego_psnr, secret_rev_psnr))
    logger.info('stego_ssim: {:.4f}, secret_rev_ssim: {:.4f}'.format(stego_ssim, secret_rev_ssim))
    logger.info('stego_lpips: {:.4f}, secret_rev_lpips: {:.4f}'.format(stego_lpips.mean().item(), secret_rev_lpips.mean().item()))
    logger.info('stego_apd: {:.2f}, secret_rev_apd: {:.2f}'.format(stego_apd, secret_rev_apd))

    stego_psnr_list.append(stego_psnr)
    secret_rev_psnr_list.append(secret_rev_psnr)
    stego_ssim_list.append(stego_ssim)
    secret_rev_ssim_list.append(secret_rev_ssim)
    stego_apd_list.append(stego_apd)
    secret_rev_apd_list.append(secret_rev_apd)
    stego_lpips_list.append(stego_lpips.mean().item())
    secret_rev_lpips_list.append(secret_rev_lpips.mean().item())


    if c.save_images:
        cover_save_path = os.path.join(image_save_dirs, 'cover', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        stego_save_path = os.path.join(image_save_dirs, 'stego', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_save_path = os.path.join(image_save_dirs, 'secret', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_rev_save_path = os.path.join(image_save_dirs, 'secret_rev', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        cover_resi_save_path = os.path.join(image_save_dirs, 'cover_resi', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_resi_save_path = os.path.join(image_save_dirs, 'secret_resi', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        mkdirs(os.path.join(image_save_dirs, 'cover'))
        mkdirs(os.path.join(image_save_dirs, 'stego'))
        mkdirs(os.path.join(image_save_dirs, 'secret'))
        mkdirs(os.path.join(image_save_dirs, 'secret_rev'))
        mkdirs(os.path.join(image_save_dirs, 'cover_resi'))
        mkdirs(os.path.join(image_save_dirs, 'secret_resi'))
        logger.info('saving images...')
        Image.fromarray(cover.astype(np.uint8)).save(cover_save_path)
        Image.fromarray(stego.astype(np.uint8)).save(stego_save_path)
        Image.fromarray(secret.astype(np.uint8)).save(secret_save_path)
        Image.fromarray(secret_rev.astype(np.uint8)).save(secret_rev_save_path)
        Image.fromarray(cover_resi.astype(np.uint8)).save(cover_resi_save_path)
        Image.fromarray(secret_resi.astype(np.uint8)).save(secret_resi_save_path)


logger.info('stego_psnr_mean: {:.2f}, stego_ssim_mean: {:.4f}, stego_lpips_mean: {:.4f}, stego_apd_mean: {:.2f}'.format(np.array(stego_psnr_list).mean(), np.array(stego_ssim_list).mean(), np.array(stego_lpips_list).mean(), np.array(stego_apd_list).mean()))
logger.info('secret_rev_psnr_mean: {:.2f}, secret_rev_ssim_mean: {:.4f}, secret_rev_lpips_mean: {:.4f}, secret_rev_apd_mean: {:.2f}'.format(np.array(secret_rev_psnr_list).mean(), np.array(secret_rev_ssim_list).mean(), np.array(secret_rev_lpips_list).mean(), np.array(secret_rev_apd_list).mean()))



