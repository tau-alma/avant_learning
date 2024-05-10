import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode

class OccupancyGridDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Path to the folder where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        self.image_size = self[0][0].shape[1]
        self.transform = Compose([
            ToTensor()  # Automatically scales to [0, 1] by dividing by 255
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # Ensure it is loaded in grayscale
        image = self.transform(image)

        labels = torch.zeros_like(image, dtype=torch.long)
        labels[image == 0.5] = 1
        labels[image == 1.0] = 2

        return image, labels.squeeze(dim=0)