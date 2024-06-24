#--
#temp for local testing
from torch.utils.data import DataLoader 
from samplers import DistributedBatchSampler
from brain import BrainDataset
import torch

import os #keep this if I keep the existing filepath strategy

#--

#keep in production
#from header import *
#from .samplers import DistributedBatchSampler
# from .sft_dataset import *
# from .mvtec import *
# from .visa import VisaDataset
# from . import all_supervised_with_cn
#from .brain import BrainDataset

'''
def get_tokenizer(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer.bos_token_id, tokenizer.eos_token_id = 1, 2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
'''

# def load_sft_dataset(args):
#     '''
#     tokenizer = get_tokenizer(args['model_path'])
#     dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
#     data_path = args["data_path"]
#     data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
#     '''
#     data = SupervisedDataset(args['data_path'], args['image_root_path'])

#     sampler = torch.utils.data.RandomSampler(data)
#     world_size = torch.distributed.get_world_size()
#     rank = torch.distributed.get_rank()
#     batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
#     batch_sampler = DistributedBatchSampler(
#         sampler, 
#         batch_size,
#         True,
#         rank,
#         world_size
#     )
#     iter_ = DataLoader(
#         data, 
#         batch_sampler=batch_sampler, 
#         num_workers=1,
#         collate_fn=data.collate, 
#         pin_memory=False
#     )
#     return data, iter_, sampler

# def load_mvtec_dataset(args):
#     '''
#     tokenizer = get_tokenizer(args['model_path'])
#     dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
#     data_path = args["data_path"]
#     data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
#     '''
#     data = MVtecDataset('../data/mvtec_anomaly_detection')

#     sampler = torch.utils.data.RandomSampler(data)
#     world_size = torch.distributed.get_world_size()
#     rank = torch.distributed.get_rank()
#     batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
#     batch_sampler = DistributedBatchSampler(
#         sampler, 
#         batch_size,
#         True,
#         rank,
#         world_size
#     )
#     iter_ = DataLoader(
#         data, 
#         batch_sampler=batch_sampler, 
#         num_workers=8,
#         collate_fn=data.collate, 
#         pin_memory=False
#     )
#     return data, iter_, sampler

def load_brain_dataset(args):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    
    #data = BrainDataset('../data/small_brain_data') 
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/small_brain_data'))
    data = BrainDataset(data_dir)

    sampler = torch.utils.data.RandomSampler(data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedBatchSampler(
        sampler, 
        batch_size,
        True,
        rank,
        world_size
    )
    iter_ = DataLoader(
        data, 
        batch_sampler=batch_sampler, 
        num_workers=8,
        collate_fn=data.collate, 
        pin_memory=False
    )
    return data, iter_, sampler


# def load_visa_dataset(args):
#     '''
#     tokenizer = get_tokenizer(args['model_path'])
#     dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
#     data_path = args["data_path"]
#     data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
#     '''
#     data = VisaDataset('../data/VisA')

#     sampler = torch.utils.data.RandomSampler(data)
#     world_size = torch.distributed.get_world_size()
#     rank = torch.distributed.get_rank()
#     batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
#     batch_sampler = DistributedBatchSampler(
#         sampler, 
#         batch_size,
#         True,
#         rank,
#         world_size
#     )
#     iter_ = DataLoader(
#         data, 
#         batch_sampler=batch_sampler, 
#         num_workers=8,
#         collate_fn=data.collate, 
#         pin_memory=False
#     )
#     return data, iter_, sampler


# def load_supervised_dataset_with_cn(args):
#     '''
#     tokenizer = get_tokenizer(args['model_path'])
#     dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
#     data_path = args["data_path"]
#     data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
#     '''
#     data = all_supervised_with_cn.SupervisedDataset('../data/all_anomalygpt')

#     sampler = torch.utils.data.RandomSampler(data)
#     world_size = torch.distributed.get_world_size()
#     rank = torch.distributed.get_rank()
#     batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
#     batch_sampler = DistributedBatchSampler(
#         sampler, 
#         batch_size,
#         True,
#         rank,
#         world_size
#     )
#     iter_ = DataLoader(
#         data, 
#         batch_sampler=batch_sampler, 
#         num_workers=1,
#         collate_fn=data.collate, 
#         pin_memory=False
#     )
#     return data, iter_, sampler