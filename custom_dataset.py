import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name

# Example usage
dataset = CustomDataset(
    root_dir='/Volumes/PHD 3.0 Silicon-Power Media/Dec 2023/OIDv4_ToolKit copy/OID/Dataset/test',
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
)
