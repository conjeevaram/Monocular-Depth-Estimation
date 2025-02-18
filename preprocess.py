import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import random

class DepthDataset(Dataset):
    def __init__(self, csv_path, base_path, transform=None, image_size=(128, 128)):
        self.base_path = base_path
        self.image_size = image_size
        self.transform = transform
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip()  #strip any extra spaces in column names from csv

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.base_path, self.data.iloc[idx]['rgb_path'])
        depth_path = os.path.join(self.base_path, self.data.iloc[idx]['depth_path'])

        # load the images
        image = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")  # depth map as grayscale (single channel)

        # resize the images
        image = image.resize(self.image_size)
        depth = depth.resize(self.image_size)

        if self.transform:
            image = self.transform(image)

        # depth normalization to [0, 1] using ToTensor
        depth = transforms.ToTensor()(depth)
        depth = transforms.Normalize(mean=[0.5], std=[0.5])(depth)

        return image, depth

# augmented dataset class
class AugmentedDataset(DepthDataset):
    def __init__(self, csv_path, base_path, transform=None, image_size=(128, 128), augment_prob=0.5):
        super().__init__(csv_path, base_path, transform, image_size)
        self.augment_prob = augment_prob

    def __getitem__(self, idx):
        image, depth = super().__getitem__(idx)
        
        # apply random augmentation with probability
        if random.random() < self.augment_prob:
            # horizontal flip
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                depth = transforms.functional.hflip(depth)
            # adjust brightness
            image = transforms.ColorJitter(brightness=0.2)(image)
        return image, depth

def get_dataloader(csv_path, base_path, batch_size=4, image_size=(128, 128), num_workers=0, augment=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize the RGB images
    ])

    dataset_class = AugmentedDataset if augment else DepthDataset
    dataset = dataset_class(csv_path=csv_path, base_path=base_path, transform=transform, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
