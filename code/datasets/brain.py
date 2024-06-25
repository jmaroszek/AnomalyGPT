import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import unittest

class BrainDataset(Dataset):
    """Dataset for brain tumor classification."""

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): Path to the root directory containing the dataset. Expects the structure to be the original
            folder structure from the kaggle dataset link. 
        """
        self.root_dir = root_dir        
        self.transform = transforms.Compose( #converts PIL image to tensor, resizes it, and normalizes it
                            [
                                transforms.ToTensor(),
                                transforms.Resize(
                                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                            ),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.labels = []

        # Iterate over each class directory and collect image paths and class labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, 'Training', class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_name)  
        #print(f'[!] Collected {len(self.image_paths)} samples for training')

    def __len__(self):
        return len(self.image_paths)

    #The _getitem__ methods for the other classes return a lot more, but I am not sure if that is needed here yet
    #if you add more here, you need to update the collate function
    def __getitem__(self, index):
        """
        Retrieve an image and its corresponding label from the dataset at the specified index.

        Args:
            index (int): Index of the image and label to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'image' (torch.Tensor): The transformed image tensor with shape (3, 224, 224).
                - 'label' (str): The class label of the image."""
        
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        return dict(image=image_tensor, label=label)

    def collate(self, items):
        """
        Custom collate function for batching samples from the BrainDataset.

        Args:
            batch (list of dict): List of samples where each sample is a dictionary 
                                containing 'image' and 'label' keys.

        Returns:
            dict: A dictionary containing:
                - 'images' (torch.Tensor): A tensor of shape (batch_size, 3, 224, 224) 
                                        representing a batch of images.
                - 'labels' (list of str): A list of labels corresponding to the batch of images.
        """
        images = [item['image'] for item in items]
        labels = [item['label'] for item in items]
        images = torch.stack(images)
        return dict(images=images, labels=labels)

class TestBrainDataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/small_brain_data'))
        self.dataset = BrainDataset(self.data_dir)

    def test_length(self):
        self.assertEqual(len(self.dataset), 20)  

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertIn('image', sample)
        self.assertIn('label', sample)
        self.assertEqual(sample['label'], 'glioma')
        self.assertEqual(sample['image'].shape, (3, 224, 224)) 

        sample_2 = self.dataset[6]
        self.assertIn('image', sample_2)
        self.assertIn('label', sample_2)
        self.assertEqual(sample_2['label'], 'meningioma')
        self.assertEqual(sample_2['image'].shape, (3, 224, 224)) 

        sample_3 = self.dataset[14]
        self.assertIn('image', sample_3)
        self.assertIn('label', sample_3)
        self.assertEqual(sample_3['label'], 'notumor')
        self.assertEqual(sample_3['image'].shape, (3, 224, 224)) 

        self.assertFalse(torch.equal(sample['image'], sample_2['image']))
        self.assertFalse(torch.equal(sample['image'], sample_3['image']))
        self.assertFalse(torch.equal(sample_2['image'], sample_3['image']))


    def test_collate(self):
        glioma_sample = self.dataset[0]
        meningioma_sample = self.dataset[6]
        num_samples = 6
        data_loader = DataLoader(self.dataset, batch_size=num_samples,collate_fn=self.dataset.collate) 

        batch = next(iter(data_loader))
        self.assertIn('images', batch)
        self.assertIn('labels', batch)
        self.assertEqual(batch['images'].shape, (num_samples, 3, 224, 224))
        self.assertEqual(len(batch['labels']), num_samples)

        self.assertTrue(batch['labels'][0] == glioma_sample['label'])
        self.assertTrue(batch['labels'][num_samples-1] == meningioma_sample['label'])

        

if __name__ == '__main__':
    unittest.main()
