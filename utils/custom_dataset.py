############################################################
#   File: custom_dataset.py                                #
#   Created: 2019-10-31 19:28:59                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:custom_dataset.py                          #
#   Copyright@2019 wvinzh, HUST                            #
############################################################

from __future__ import print_function, division

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
import random
import pandas


class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, tokenizer=None):

        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.data = pd.read_csv(txt_file, sep='\t')
        self.data = self.data.dropna(axis=0, how='any')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data.iloc[idx]
        label = d['label']
        img_name = d['file_path']
        text = d['text']
        img_file = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_file).convert('RGB')
        except Exception as e:
            print(d)

        if self.transform:
            image = self.transform(image)
        text_input = self.tokenizer(text,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
                )


        return image, torch.tensor(label), text_input


def test_dataset():
    root = '/home/zengh/Dataset/Fine-grained/CUB_200_2011/images'
    txt = '/home/zengh/Dataset/Fine-grained/CUB_200_2011/test_pytorch.txt'
    from torchvision import transforms
    rgb_mean = [0.5, 0.5, 0.5]
    rgb_std = [0.5, 0.5, 0.5]
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    carData = CustomDataset(txt, root, transform_val, True)
    print(carData.num_classes)
    dataloader = DataLoader(carData, batch_size=16, shuffle=True)
    for data in dataloader:
        images, labels = data
        # print(images.size(),labels.size(),labels)


if __name__ == '__main__':
    test_dataset()
