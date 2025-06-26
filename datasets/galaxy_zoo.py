import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GalaxyZooDataset(Dataset):
    """
    Custom PyTorch Dataset for the Galaxy Zoo 2 dataset.
    This class assumes you have a CSV file with galaxy information and a
    directory containing the corresponding images.
    
    The main dataset can be found at: https://data.galaxyzoo.org/
    A common setup involves downloading images based on the CSV catalog.
    """
    def __init__(self, data_dir, csv_path, transform=None, label_column='gz2_class'):
        """
        Args:
            data_dir (string): Directory with all the images.
            csv_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            label_column (string): The name of the column in the CSV file to use as labels.
                                   'gz2_class' is a common choice, containing classifications
                                   like 'SPIRAL', 'ELLIPTICAL', etc.
        """
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column

        # Create a mapping from string labels to integer indices
        self.classes = self.df[self.label_column].unique()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Assumes an 'asset_id' or similar column for the image name
        img_name = os.path.join(self.data_dir, f"{row.name}.jpg") 
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Handle cases where an image might be missing
            print(f"Warning: Image not found at {img_name}. Skipping.")
            # Return a placeholder or skip. For simplicity, we'll return None.
            # A more robust implementation might try the next valid index.
            return None, None

        label_name = row[self.label_column]
        label_idx = self.class_to_idx[label_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx 