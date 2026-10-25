import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict
from PIL import Image
import os
import torchvision.transforms.functional as functional
from consts import *

classMap = {
    0: 0,       # BG
    1: 1,       # Caption jako Text
    2: 1,       # Footnote jako Text
    3: 2,       # Formula
    4: 1,       # List item jako Text
    5: 1,       # Page Footer jako Text
    6: 1,       # Page header jako Text
    7: 3,       # Picture
    8: 4,       # Section Header jako Title
    9: 5,       # Table
    10: 1,      # Text
    11: 4       # Title
}

class DocLayNetDataset(Dataset):
    def __init__(self, filename):
        with open(filename, "r") as file:
            data = json.load(file)
        self.images = data["images"]
        annotations = data["annotations"]
        self.img_annotations = defaultdict(list)
        for ann in annotations:
            self.img_annotations[ann["image_id"]].append(ann)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        filename = self.images[id]["file_name"]
        image_id = self.images[id]["id"]
        image = Image.open(os.path.join(IMG_SUBFOLDER, filename)).convert("RGB")
        w, h = image.size
        image = image.resize((SIZE, SIZE))
        image = 1.0 - functional.to_tensor(image)
        target = torch.zeros(SIZE, SIZE, dtype=torch.long)
        xScale, yScale = (SIZE/w, SIZE/h)
        for ann in self.img_annotations[image_id]:
            x, y, w, h = ann["bbox"]
            x, y, w, h = int(x*xScale), int(y*yScale), int(w*xScale), int(h*yScale)
            y2 = min(y+h, SIZE)
            x2 = min(x+w, SIZE)
            target[y:y2, x:x2] = classMap[ann["category_id"]]
        return image, target


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.cnn(x)


class UNETBasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(p=0.3)
        self.maxpool = nn.MaxPool2d((2,2))

        self.encoderI = nn.Sequential(
            ConvBlock(3, 32),
        )
        self.encoderII = nn.Sequential(
            ConvBlock(32, 64),
        )
        self.encoderIII = nn.Sequential(
            ConvBlock(64, 128),
        )
        self.encoderIV = nn.Sequential(
            ConvBlock(128, 256),
        )

        self.bottom = nn.Sequential(
            ConvBlock(256, 512),
            self.dropout
        )

        self.decoderUpIV = nn.ConvTranspose2d(512,256,2, stride=2)
        self.decoderConvIV = nn.Sequential(
            ConvBlock(512, 256),
            self.dropout
        )
        self.decoderUpIII = nn.ConvTranspose2d(256,128,2, stride=2)
        self.decoderConvIII = nn.Sequential(
            ConvBlock(256, 128),
        )
        self.decoderUpII = nn.ConvTranspose2d(128,64,2, stride=2)
        self.decoderConvII = nn.Sequential(
            ConvBlock(128, 64),
        )
        self.decoderUpI = nn.ConvTranspose2d(64,32,2, stride=2)
        self.decoderConvI = nn.Sequential(
            ConvBlock(64, 32),
        )
        self.head = nn.Conv2d(32, CLASSES, kernel_size=1, stride=1)
    
    def forward(self, x):
        # 512x512
        e1 = self.encoderI(x)
        e2 = self.encoderII(self.maxpool(e1))
        # 256x256
        e3 = self.encoderIII(self.maxpool(e2))
        # 128x128
        e4 = self.encoderIV(self.maxpool(e3))
        # 64x64

        b = self.bottom(self.maxpool(e4))
        # 32x32

        d4 = self.decoderUpIV(b)
        d4 = self.decoderConvIV(torch.cat([d4, e4], dim=1))
        # 64x64
        d3 = self.decoderUpIII(d4)
        d3 = self.decoderConvIII(torch.cat([d3, e3], dim=1))
        # 128x128
        d2 = self.decoderUpII(d3)
        d2 = self.decoderConvII(torch.cat([d2, e2], dim=1))
        # 256x256
        d1 = self.decoderUpI(d2)
        d1 = self.decoderConvI(torch.cat([d1, e1], dim=1))
        # 512x512

        return self.head(d1)

