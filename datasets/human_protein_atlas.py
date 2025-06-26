import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HumanProteinAtlasDataset(Dataset):
    """
    Custom PyTorch Dataset for the Human Protein Atlas dataset.
    Assumes data is downloaded from the Kaggle competition:
    https://www.kaggle.com/c/human-protein-atlas-image-classification
    """
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['Id']
        
        # The images are multi-channel (Red, Green, Blue, Yellow)
        # We will map R, G, B to the RGB channels for this example
        colors = ['red', 'green', 'blue']
        img_channels = []
        for color in colors:
            img_path = os.path.join(self.data_dir, f"{img_id}_{color}.png")
            # Open as grayscale and convert to numpy array
            channel = np.array(Image.open(img_path).convert('L'))
            img_channels.append(channel)

        # Stack channels to form an RGB image
        img_np = np.stack(img_channels, axis=-1)
        img = Image.fromarray(img_np, 'RGB')
        
        # Labels are multi-label, encoded as space-separated strings
        labels = list(map(int, row['Target'].split()))
        
        # For this diffusion model, we might use a single class or a specific
        # combination as the conditioning. Here, we simplify by taking the first label.
        label = labels[0]

        if self.transform:
            img = self.transform(img)
            
        return img, label 