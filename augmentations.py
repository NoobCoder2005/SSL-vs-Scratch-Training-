import torchvision.transforms as transforms
import random
import torch
import cv2
import numpy as np

# --- Custom Transform: Gaussian Noise ---
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


# --- Custom Transform: CLAHE (for CT contrast) ---
class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = clahe.apply(img)
        img = img.astype(np.float32) / 255.0
        return img


# --- SimCLR Augmentations ---
def get_simclr_augmentations():

    return transforms.Compose([

        # Convert to PIL (needed for torchvision ops)
        transforms.ToPILImage(),

        # Slight geometric transforms (SAFE)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),

        # Intensity augmentations (VERY IMPORTANT)
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.8),

        # Optional CLAHE (good for CT contrast); use class not Lambda so DataLoader workers can pickle.
        CLAHE(),

        # Back to tensor
        transforms.ToTensor(),

        # Add noise
        AddGaussianNoise(mean=0.0, std=0.02),

        # Slight blur
        transforms.GaussianBlur(kernel_size=3)
    ])