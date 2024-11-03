#-*- coding: utf-8 -*-

import os
import sys
import math
import time
import torch
import random
import threading
import logging
import librosa
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
from warnings import warn
from torch.utils.data.sampler import Sampler

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

random_seed = 2891

import torch
import torch.nn as nn
import torch.optim as optim
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

PAD = 0

np.seterr(divide = 'ignore')
np.seterr(divide = 'warn')

torchaudio.set_audio_backend("sox_io")
        
class StaticFeatureDataset(Dataset):
    def __init__(self, path_list):
        super(StaticFeatureDataset, self).__init__()
        
        self.wav_list = path_list
        self.size = len(self.wav_list)        
        

    def __getitem__(self, index):
        try:
            feature = np.load(self.wav_list[index])
            
        except:
            print('data {} has problem, can not reading '.format(self.wav_list[index]))
            return None
        
        #feature = torch.FloatTensor(feature).permute(1, 0) # [D, T] -> [T, D]
        feature = torch.FloatTensor(feature) # [D, T]
        
        if feature.size(-1) < 50:
            return None
        
        if feature.size(-1) > 1500:
            feature = feature[:, :1500]
                
        return feature
        
    def __len__(self):
        return self.size

'''
def _collate_fn(batch):    
    batches = list(filter(lambda x: x is not None, batch)) 
    batch = sorted(batches, key=lambda sample: sample[0].size(1), reverse=True)
    seq_lengths    = [s[0].size(1) for s in batch]
    
    max_seq_size = max(seq_lengths)

    feat_size = batch[0][0].size(0)
    batch_size = len(batch)
    
    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    
    
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_length = tensor.size(1)
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)

    return seqs
'''

def _collate_fn(batch):    
    # Filter out None entries
    batches = list(filter(lambda x: x is not None, batch))
    
    if len(batches) == 0:
        return None  # Return None if all elements are None

    # Calculate sequence lengths and maximum sequence size
    seq_lengths = [s.size(1) for s in batches]    
    max_seq_size = max(seq_lengths)
    
    # Get the feature dimension and batch size after filtering
    feat_size = batches[0].size(0)    
    batch_size = len(batches)

    # Initialize the tensor for the batch with filtered size
    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    
    # Populate the batch tensor with valid data
    for x in range(batch_size):                
        tensor = batches[x]
        seq_length = tensor.size(1)
        seqs[x][0, :, :seq_length].copy_(tensor)
    
    return seqs

    
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

class AudioDataLoaderForTest(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoaderForTest, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_test



class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
        



############ Dynamic Feature Extractor    

class DynamicFeatureDataset(Dataset):
    def __init__(self, path_list):
        super(DynamicFeatureDataset, self).__init__()
        
        self.path_list = path_list
        self.size = len(self.path_list)

    def __getitem__(self, index):
        feature_path = self.path_list[index]
        feature, name = self.parse_feature(feature_path)        
        return feature, name

    def parse_feature(self, feature_path):
        file_name = os.path.basename(feature_path)        
        feature = np.load(feature_path)
        feature = torch.FloatTensor(feature)
        
        return feature, file_name
    
    def __len__(self):
        return self.size

# just only one batch

def _collate_feature_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(-1), reverse=True)
    seq_lengths    = [s[0].size(-1) for s in batch]
    
    max_seq_size = max(seq_lengths)
    feat_size = batch[0][0].size(0)
        
    batch_size = len(batch)
    

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = list()
    
    for x in range(batch_size):        
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        
        seq_length = tensor.size(-1)
        
        #seqs[x].narrow(0, 0, len(tensor)).copy_(tensor)
        seqs[x][0, :, :seq_length].copy_(tensor)
        targets.append(target)

    #seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets
    
    

    
class FeatureDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(FeatureDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_feature_fn