import torch
import torch.nn as nn
import editdistance
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, random_split
import os
import random
import pickle
import numpy as np
from zdania.utils.model import *
from zdania.utils.consts import *

# Ładowanie danych 
class OCRDataset(Dataset):

    def __init__(self, mytransforms=None):
        self._transforms = mytransforms or []
        self._load_dataset()
        self._arrange_dataset()
        self._prepareEncoding()
        self._transforms = transforms.Compose([
            *self._transforms,
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.paths)

    def _prepareEncoding(self):
        self.idx_to_char = {
            0: "<BLANK>"
        }
        for id, char in enumerate(sorted(self.vocab), start=1):
            self.idx_to_char[id] = char
        self.char_to_idx = {char:idx for idx, char in self.idx_to_char.items() if char != "<BLANK>"}

    def _load_dataset(self):
        self.dataset, self.vocab, self.maxLength = [], set(), 0
        words = open("words.txt", "r").readlines()
        for line in words:
            splitS = line.split(" ----- ")
            file = splitS[1].rstrip('\n')
            label = splitS[0].rstrip('\n')
            self.dataset.append([file, label])
            self.vocab.update(list(label))
            if len(label)>self.maxLength:
                self.maxLength = len(label)

    def _arrange_dataset(self):
        self.paths = np.array([x[0] for x in self.dataset], dtype=str)
        self.labels = np.array([x[1] for x in self.dataset], dtype=str)

    def split(self, rate):
        train_size = int(rate * self.__len__())
        val_size = self.__len__() - train_size
        return random_split(self, [train_size, val_size])

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        label = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        image = Image.open(path).convert("RGB")
        image = resizeWithPad(image)
        image = self._transforms(image)
        return image, label


# Model

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.block(x)

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            # Wzor: Out = floor((In + 2Pad - Ker)/Stride) + 1
            # 1024x32
            ConvBlock(3,16),
            ConvBlock(16,32),
            # 1024 x 32
            nn.MaxPool2d((2,2)),
            # 512 x 16
            ConvBlock(32,64),
            ConvBlock(64,96),
            # 512 x 16
            ConvBlock(96,128),
            ConvBlock(128,256),
            nn.MaxPool2d((2,2)),
            # 256 x 8
            ConvBlock(256,512),
            
        )
        self.blstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )
        self.blstm2 = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self,x):
        x = self.cnn(x)
        x = self.pool(x)
        batch, channels, height, width = x.shape
        x = x.permute(0,3,1,2)
        x = x.reshape(batch, width, channels*height)
        # BLSTM dostaje: batch, sequence_width, sequence_parameters
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x, _ = self.blstm2(x)
        x = self.dropout(x)
        x = self.fc(x)
        # Funkcja liniowa zwraca: batch, sequence_width, klasy (litery)
        return x
