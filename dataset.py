import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MontgomeryDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.endswith(".png") and os.path.exists(os.path.join(mask_dir, f.replace(".png", "_mask.png")))
        ]

        for f in os.listdir(img_dir):
            if f.endswith(".png") and not os.path.exists(os.path.join(mask_dir, f.replace(".png", "_mask.png"))):
                print(f"Warning: Mask not found for {f} at {os.path.join(mask_dir, f.replace('.png', '_mask.png'))}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file.replace(".png", "_mask.png"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]        # tensor shape: [3, 512, 512]
            mask = augmented["mask"].unsqueeze(0).float()  # shape: [1, 512, 512]

        return image, mask

    
def get_transforms(train=True, image_size=512):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])    


if __name__ == "__main__":
    # Test the dataset with COMBINED masks
    dataset = MontgomeryDataset(
        img_dir=os.path.normpath(r"C:\Users\danie\OneDrive\Desktop\INM705\archive\Montgomery\MontgomerySet\CXR_png"),
        mask_dir=os.path.normpath(r"C:\Users\danie\OneDrive\Desktop\INM705\archive\Montgomery\MontgomerySet\masks")
    )
    print(f"Found {len(dataset)} valid pairs")
    if len(dataset) > 0:
        img, mask = dataset[0]

        # Convert PIL images to tensors for shape inspection
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img)
        mask_tensor = to_tensor(mask)

        print(f"Image shape: {img_tensor.shape}, Mask shape: {mask_tensor.shape}")