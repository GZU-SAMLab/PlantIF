import os
import pickle
import numpy as np
from random import random

from transformers import BertTokenizer
from torchvision import transforms

from config import get_config, activation_dict
# from data_loader import get_loader
from solver import Solver
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
sys.path.append(os.getcwd())
from utils.custom_dataset import CustomDataset

if __name__ == '__main__':
    
    # Setting random seed
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    print(train_config)

    # Creating pytorch dataloaders
    # train_data_loader = get_loader(train_config, shuffle = True)
    # dev_data_loader = get_loader(dev_config, shuffle = False)
    # test_data_loader = get_loader(test_config, shuffle = False)


    rgb_mean = [0.5, 0.5, 0.5]
    rgb_std = [0.5, 0.5, 0.5]
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')

    # train_dataset = CustomDataset('CUB_200_2011/bird_train.txt', 'CUB_200_2011/images', transform_val, tokenizer)
    # dev_dataset = CustomDataset('CUB_200_2011/bird_test.txt', 'CUB_200_2011/images', transform_val, tokenizer)

    train_dataset = CustomDataset("/home/wuxingcai/pretrain-leaf/new/leaf_diseases/train.csv", "/home/wuxingcai/pretrain-leaf/new/leaf_diseases/", transform_val, tokenizer)
    dev_dataset = CustomDataset("/home/wuxingcai/pretrain-leaf/new/leaf_diseases/test.csv", "/home/wuxingcai/pretrain-leaf/new/leaf_diseases/", transform_val, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    dev_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_loader, dev_loader, dev_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    solver.train()
