import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class BrainDataset(Dataset):
    """Dataset for brain tumor classification."""

    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.labels = []

        # Iterate over each class directory and collect image paths and labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, 'Training', class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label) #label matches folder name 
        print(f'[!] Collected {len(self.image_paths)} samples for training')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return dict(image=image, label=label)

    def collate(self, instances):
        images = [instance['image'] for instance in instances]
        labels = [instance['label'] for instance in instances]
        
        images = torch.stack(images)
        labels = torch.tensor(labels)

        return dict(images=images, labels=labels)
