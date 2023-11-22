import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch
import os
from PIL import Image

import glob

class dataset_(Dataset):
    def __init__(self, cover_img_dir, stego_img_dir, transform):
        self.cover_img_dir = cover_img_dir
        self.stego_img_dir = stego_img_dir
        self.transforms = transform
        self.cover_img_filenames = list(sorted(os.listdir(cover_img_dir)))
        self.stego_img_filenames = list(sorted(os.listdir(stego_img_dir)))
    
    def __len__(self):
        return len(self.cover_img_filenames)
    
    def __getitem__(self, index):
        cover_img_paths = os.path.join(self.cover_img_dir, self.cover_img_filenames[index])
        # print(cover_img_paths)
        stego_img_paths = os.path.join(self.stego_img_dir, self.stego_img_filenames[index])
        # print(stego_img_paths)
 
        cover_img = Image.open(cover_img_paths).convert("RGB")
        stego_img = Image.open(stego_img_paths).convert("RGB")
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)

        label1 = torch.tensor(0, dtype=torch.long)
        label2 = torch.tensor(1, dtype=torch.long)

        sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [label1, label2]

        return sample
    

transform_train = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomRotation(degrees=90),
    T.ToTensor()
])

transform_val = T.Compose([
    T.ToTensor(),
])

transform_train = transform_val

def load_train_data(cover_data_dir, stego_data_dir, batchsize=4,):

    train_loader = DataLoader(
        dataset_(cover_data_dir, stego_data_dir, transform_train),
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True,
        # num_workers=8,
        drop_last=True
    )

    return train_loader

def load_test_data(cover_data_dir, stego_data_dir, batchsize=4,):

    test_loader = DataLoader(
        dataset_(cover_data_dir, stego_data_dir, transform_val),
        batch_size=batchsize,
        shuffle=True,
        pin_memory=False,
        # num_workers=8,
        drop_last=False
    )

    return test_loader


# transform_train = A.Compose(
#     [
#         A.RandomCrop(128, 128),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         ToTensorV2(),
#     ]
# )

# transform_val = A.Compose([
#     A.CenterCrop(256, 256),
#     ToTensorV2(),
# ])


# class dataset_(Dataset):
#     def __init__(self, img_dir, sigma, transform):
#         self.img_dir = img_dir
#         self.img_filenames = list(sorted(os.listdir(img_dir)))
#         self.sigma = sigma
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_filenames)

#     def __getitem__(self, idx):
#         img_filename = self.img_filenames[idx]
#         img = cv2.imread(os.path.join(self.img_dir, img_filename))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img/255)
        
#         if self.transform:
#             img = self.transform(image=img)["image"]
            
#         noised_img = img + torch.randn(img.shape).mul_(self.sigma/255)

#         return img, noised_img


# def load_dataset(train_data_dir, test_data_dir, batch_size, sigma=None):

#     train_loader = DataLoader(
#         dataset_(train_data_dir, sigma, transform_train),
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=8,
#         drop_last=True
#     )

#     test_loader = DataLoader(
#         dataset_(test_data_dir, sigma, transform_val),
#         batch_size=2,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=1,
#         drop_last=True
#     )

#     return train_loader, test_loader


    
