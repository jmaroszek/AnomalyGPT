import unittest
import torch
from torch.utils.data import DataLoader
from brain import BrainDataset  
from __init__ import load_brain_dataset
from samplers import DistributedBatchSampler

class TestLoadBrainDataset(unittest.TestCase):
    def setUp(self):
        # Mocking distributed training environment
        self.original_world_size = torch.distributed.get_world_size
        self.original_rank = torch.distributed.get_rank
        torch.distributed.get_world_size = lambda: 1
        torch.distributed.get_rank = lambda: 0

        # Mock arguments
        self.args = {
            'world_size': 1,
            'dschf': type('', (), {})()  # Dummy object for dschf
        }
        self.args['dschf'].config = {'train_micro_batch_size_per_gpu': 2}
        
        
    def tearDown(self):
        torch.distributed.get_world_size = self.original_world_size
        torch.distributed.get_rank = self.original_rank

    def test_load_brain_dataset(self):
        data, iter_, sampler = load_brain_dataset(self.args)
        self.assertIsInstance(data, BrainDataset)
        self.assertIsInstance(iter_, DataLoader)
        self.assertIsInstance(sampler, DistributedBatchSampler) #currently failing here and below
        self.assertEqual(iter_.batch_size, 2)
        self.assertEqual(len(iter_), len(data) // 2)

        # Check if the DataLoader loads data correctly
        batch = next(iter(iter_))
        self.assertIn('images', batch)
        self.assertIn('labels', batch)
        self.assertEqual(batch['images'].shape[1:], (3, 224, 224))  # Check image shape
        self.assertEqual(len(batch['labels']), 2)  # Check label batch size

if __name__ == '__main__':
    unittest.main()
