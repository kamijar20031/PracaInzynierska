import torch
import torch.nn as nn
from consts import *


def ctc_decode(sequence, blank=0):
    decoded = []
    previous = None
    for token in sequence:
        token = int(token)
        if token == blank:
            previous = token
            continue
        if token == previous:
            continue
        decoded.append(token)
        previous = token
    return decoded

def decode_text(tokens, idx_to_char):
    chars = []
    for token in tokens:
        chars.append(idx_to_char[token])
    return "".join(chars)

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
            # 512x32
            ConvBlock(3,16),
            ConvBlock(16,32),
            # 512 x 32
            nn.MaxPool2d((2,2)),
            # 256 x 16
            ConvBlock(32,64),
            ConvBlock(64,96),
            # 256 x 16
            ConvBlock(96,128),
            ConvBlock(128,256),
            nn.MaxPool2d((2,2)),
            # 128 x 8
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
