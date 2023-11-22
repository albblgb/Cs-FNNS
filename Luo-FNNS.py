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
import kornia

from models.decodingNetwork import dec_img, stegan_dec, decodingNetwork
from models.network_dncnn import DnCNN
from utils.model import init_weights, shuffle_params
from utils.image import calculate_ssim, calculate_psnr, calculate_mae
from utils.logger import logging, logger_info
from utils.dir import mkdirs
from utils.draw import img_hist
import config as c


mode = "random"
steps = 50
iter1 = 15
iter2 = 15
alpha = 0.1
beta = 0 # since we use the random decoding networks, keys are not necessary,  as the seed for init the decoing network acts as keys shared between the sender and receiver.
mu1 = 1
mu2 = 1
rho1 = 0.5
rho2 = 0.5
num_bits = 1

secret_dataset = c.secret_dataset_dir.split('/')[-2]
cover_dataset = c.cover_dataset_dir.split('/')[-2]

logger_name = 'Luo-FNNS'
image_save_dirs = os.path.join('results', logger_name, secret_dataset)
mkdirs(image_save_dirs)
logger_info(logger_name, log_path=os.path.join(image_save_dirs, 'result' + '.log'))
logger = logging.getLogger(logger_name)
logger.info('secret dataset: {:s}'.format(secret_dataset))
logger.info('cover dataset: {:s}'.format(cover_dataset))
logger.info('learning rate: {:.3f}'.format(alpha))
# logger.info('epsilon: {:.2f}'.format(eps))
logger.info('number of iterations: {}'.format(steps))
logger.info('the size of secret image: {}'.format(c.secret_image_size))
logger.info('the size of cover image: {}'.format(c.cover_image_size))
os.environ["CUDA_VISIBLE_DEVICES"] = c.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


criterion = torch.nn.MSELoss()
def pixel_loss_cal(input1, input2, mask):
    loss_fn = torch.nn.MSELoss(reduction='none')
    # loss = torch.sum(loss_fn(input1, input2) * mask, dim=(1,2,3))
    loss = torch.mean(loss_fn(input1, input2) * mask, dim=(1,2,3))
    return loss.cuda()
criterion1 = torch.nn.MSELoss()


if c.cover_image_size // c.secret_image_size == 1:
    down_ratio_l3 = 1; down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 2:
    down_ratio_l3 = 2; down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 4:
    down_ratio_l3 = 2; down_ratio_l2 = 2
else:
    print('The code does not take into account the current situation, please adjust the image resulation')


import PerceptualSimilarity.models
# parparing decoder and denosing model
model = decodingNetwork(input_channel=3*c.psf*c.psf, output_channels=3*c.psf*c.psf, down_ratio_l2=down_ratio_l2, down_ratio_l3=down_ratio_l3).to(device)
denoise_model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R').to(device)
denoise_model.load_state_dict(torch.load('models/dncnn_color_blind.pth'), strict=True)
LpipsNet = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0]) # For calculating LPIPS


secret_image_path_list = list(sorted(glob.glob(os.path.join(c.secret_dataset_dir, '*'))))
cover_image_path_list = list(sorted(glob.glob(os.path.join(c.cover_dataset_dir, '*'))))


stego_psnr_list = []; stego_ssim_list = []; stego_lpips_list = []; stego_apd_list = []
secret_rev_psnr_list = []; secret_rev_ssim_list = []; secret_rev_lpips_list = []; secret_rev_apd_list = []


class Cost_Mask(nn.Module):
    def __init__(self):
        super(Cost_Mask, self).__init__()
        self.HF = torch.tensor([[[-1,2,-1],[2,-4,2],[-1,2,-1]]])
        self.H2 = torch.full((1,3,3), 1/(3*3), dtype=torch.float32)
        self.HW = torch.full((1,15,15), 1/(15*15), dtype=torch.float32)

    def cal_loss(self, image):
        sizeImage = image.size()
        padsize = max(self.HF.size())
        imagePadded = torch.tensor(np.lib.pad(image.cpu().numpy(), ((0,0),(0,0),(padsize,padsize),(padsize,padsize)), 'symmetric'), dtype=torch.float64)
        R = kornia.filters.filter2d(imagePadded, self.HF, border_type='constant', padding='same')
        W = kornia.filters.filter2d(torch.abs(R), self.H2, border_type='constant', padding='same')
        # remove padding
        sizeW = W.size()
        W = W[:, :, (sizeW[2]-sizeImage[2])//2:sizeW[2]-padsize, (sizeW[3]-sizeImage[3])//2:sizeW[3]-padsize]
        cost = 1.0 / (W+1e-10)
        wetCost = 1e10
        # compute embedding costs
        rhoA = cost
        rhoA = torch.where(rhoA > wetCost, wetCost, rhoA)
        rhoA = torch.where(torch.isnan(rhoA), wetCost, rhoA)
        # cost = kornia.filters.filter2d(rhoA, self.HW, border_type='reflect', padding='same')
        padsize_1 = max(self.HW.size())//2
        imagePadded_1 = torch.tensor(np.lib.pad(rhoA.cpu().numpy(), ((0,0),(0,0),(padsize_1,padsize_1),(padsize_1,padsize_1)), 'symmetric'), dtype=torch.float64)
        cost = kornia.filters.filter2d(imagePadded_1, self.HW, border_type='constant', padding='same')
        # remove padding
        sizeCost = cost.size()
        cost = cost[:, :, (sizeCost[2]-sizeImage[2])//2:sizeCost[2]-padsize_1, (sizeCost[3]-sizeImage[3])//2:sizeCost[3]-padsize_1]
        return cost

    def forward(self, image):
        wetCost = 1e8
        rho = self.cal_loss(image)
        # adjust embedding costs
        rho = torch.where(rho > wetCost, wetCost, rho) #threshold on the costs
        rho = torch.where(torch.isnan(rho), wetCost, rho)  #if all xi{} are zero threshold the cost
        rho = torch.where(rho > 0.5, 3., rho)
        mask = rho.to(dtype=torch.float32)
        if mask.size()[1] == 1:
            mask = mask.repeat(1,3,1,1)
        return mask


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

    # key = (torch.bernoulli(torch.empty(1, 3, cover.shape[1], cover.shape[0]).uniform_(0, 1)).to('cuda') - 0.5) / 50
    key = torch.zeros_like(cover)
    # key_random = (torch.bernoulli(torch.empty(1, 3, cover.shape[2], cover.shape[3]).uniform_(0, 1)).to('cuda') - 0.5) / 50
    # key_random_1 = (torch.bernoulli(torch.empty(1, 3, cover.shape[2], cover.shape[3]).uniform_(0, 1)).to('cuda') - 0.5) / 50
    # key_random_2 = (torch.bernoulli(torch.empty(1, 3, cover.shape[2], cover.shape[3]).uniform_(0, 1)).to('cuda') - 0.5) / 50
    # key_random_3 = (torch.bernoulli(torch.empty(1, 3, cover.shape[2], cover.shape[3]).uniform_(0, 1)).to('cuda') - 0.5) / 50
    # key_random_4 = (torch.bernoulli(torch.empty(1, 3, cover.shape[2], cover.shape[3]).uniform_(0, 1)).to('cuda') - 0.5) / 50
    # key_random_5 = (torch.bernoulli(torch.empty(1, 3, cover.shape[2], cover.shape[3]).uniform_(0, 1)).to('cuda') - 0.5) / 50

    # praparing decoder
    random_seed_for_decodor = random.randint(0, 100000000)
    logger.info('random_seed_for_decodor(receiver): {:s}'.format(str(random_seed_for_decodor)))
    init_weights(model, random_seed_for_decodor)
    # model.apply(shuffle_params)
    model = model.to(device)
    model.eval()
    
    Cmask = Cost_Mask()
    mask = 1. + Cmask(cover*255).cuda()

    cover_copy = cover.clone().detach().contiguous()
    delta = torch.zeros_like(cover_copy).cuda()

    for j in range(steps):
        delta.requires_grad = True
        optimizer = torch.optim.LBFGS([delta], lr=alpha, max_iter=iter1)

        def closure():
            adv_image = cover_copy + delta
            key_image = adv_image + key
            
            outputs2 = model(key_image)

            pixel_loss = pixel_loss_cal(adv_image, cover, mask)
        
            loss = mu1 * criterion(outputs2, secret) + rho1 * pixel_loss

            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)

        optimizer = torch.optim.LBFGS([delta], lr=alpha, max_iter=iter2)

        def closure():
            adv_image = cover_copy + delta
            key_image = adv_image + key

            outputs1 = model(adv_image)
            outputs2 = model(key_image)
            # outputs_error_1 = model(adv_image + key_random_1)
            # outputs_error_2 = model(adv_image + key_random_2)
            # outputs_error_3 = model(adv_image + key_random_3)
            # outputs_error_4 = model(adv_image + key_random_4)
            # outputs_error_5 = model(adv_image + key_random_5)
            pixel_loss = pixel_loss_cal(adv_image, cover, mask)
            loss = mu2 * criterion(outputs2, secret) + rho2 * pixel_loss
            # loss = mu2 * criterion(outputs2, secret) + rho2 * pixel_loss \
            #     +  beta * criterion(outputs1, secret)  + beta * criterion(outputs_error_1, secret) + beta * criterion(outputs_error_2, secret) + beta * criterion(outputs_error_3, secret) \
            #     + 0 * criterion(outputs_error_4, secret) + 0 * criterion(outputs_error_5, secret) 
            
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)

        # adv_image = torch.clamp(cover + delta, min=0, max=1)
        # adv_image = torch.round(adv_image*255) / 255
        # delta = (adv_image - cover).detach().contiguous()

    # rounding and clipping operations
    adv_image = torch.round(torch.clamp((cover + delta)*255, min=0., max=255.))/255

    # testing 
    secret_rev = model(adv_image)
    secret_resi_draw = (secret - secret_rev)
    secret_resi_draw = secret_resi_draw.flatten().detach().squeeze().cpu().numpy()*255
    hist_save_dir = os.path.join(image_save_dirs, 'resi/')
    mkdirs(hist_save_dir)
    img_hist(secret_resi_draw, os.path.join(hist_save_dir +  str(i) + '_BP.png'), tilte='Luo')
    # denosing the recovered secret images
    secret_rev = denoise_model(secret_rev)
    secret_rev = torch.round(torch.clamp(secret_rev*255, min=0., max=255.))/255
    cover_resi = (adv_image - cover).abs() * c.resi_magnification
    secret_resi = (secret_rev - secret).abs() * c.resi_magnification

    # calculing lpips
    stego_lpips = LpipsNet.forward(cover, adv_image, normalize=True)
    secret_rev_lpips = LpipsNet.forward(secret, secret_rev, normalize=True)


    # tensor(cuda) to numpy(cpu)
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



