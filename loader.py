from PIL import Image
import random
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class SRDataset(Dataset):
    def __init__(self, div2k_path, patch_size=48):
        self.hr_images_path = os.path.join(div2k_path, 'DIV2K_train_HR')
        self.lr_images_path = os.path.join(div2k_path, 'DIV2K_train_LR_bicubic/X4')
        self.hr_images = sorted([img for img in os.listdir(self.hr_images_path) if img.endswith('.png')])
        self.lr_images = sorted([img for img in os.listdir(self.lr_images_path) if img.endswith('.png')])
        assert len(self.hr_images) == len(self.lr_images)
       
        self.patch_size = patch_size
        self.scale = 4

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_images_path, self.hr_images[idx])).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_images_path, self.lr_images[idx])).convert('RGB')

        # 이미지 중심 좌표
        hr_width = hr_img.size[0]
        hr_height = hr_img.size[1]

        # 중앙 크롭을 위한 좌표 계산
        x = (hr_width - self.patch_size) // 2
        y = (hr_height - self.patch_size) // 2

        # HR 이미지 센터 크롭
        hr_patch = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # LR 이미지 센터 크롭 (scale factor 고려)
        lr_patch = lr_img.crop((x//self.scale, y//self.scale,
                              (x + self.patch_size)//self.scale,
                              (y + self.patch_size)//self.scale))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        hr_patch = transform(hr_patch)
        lr_patch = transform(lr_patch)

        return lr_patch, hr_patch