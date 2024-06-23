import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import unittest

class BrainDataset(Dataset):
    """Dataset for brain tumor classification."""

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transforms.Resize(
                                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.labels = []

        # Iterate over each class directory and collect image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, 'Training', class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_name)  
        print(f'[!] Collected {len(self.image_paths)} samples for training')

    def __len__(self):
        return len(self.image_paths)

    #should this return an image or a tensor?
    #not sure if this does what the other files do
    # it seems they are returning a lot more than I am (mvtech, visa, )
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

class TestBrainDataset(unittest.TestCase):
    def setUp(self):
        # Define a small dataset directory structure for testing
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/small_brain_data'))
        print(f"Data directory: {self.data_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = BrainDataset(self.data_dir, transform=self.transform)

    def test_length(self):
        # Adjust the expected length based on your test data
        self.assertEqual(len(self.dataset), 20)  

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertIn('image', sample)
        self.assertIn('label', sample)
        self.assertEqual(sample['image'].shape, (3, 224, 224)) #Image class doesn't have this method. should it be a tensor or image?
        self.assertIsInstance(sample['label'], int)

    def test_collate(self):
        sample1 = self.dataset[0]
        sample2 = self.dataset[1]
        collated = self.dataset.collate([sample1, sample2])
        self.assertIn('images', collated)
        self.assertIn('labels', collated)
        self.assertEqual(collated['images'].shape, (2, 3, 224, 224))
        self.assertEqual(collated['labels'].shape, (2,))

if __name__ == '__main__':
    unittest.main()
