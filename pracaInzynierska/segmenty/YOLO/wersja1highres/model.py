import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict
from PIL import Image
import os
import torchvision.transforms.functional as functional
from consts import *


def intersectionOverUnion(pCoord, pSize, tCoord, tSize):
    left = torch.max(pCoord[..., 0] - pSize[..., 0]/2, tCoord[..., 0] - tSize[..., 0]/2)
    top = torch.max(pCoord[..., 1] - pSize[..., 1]/2, tCoord[..., 1] - tSize[..., 1]/2)
    bottom =  torch.min(pCoord[..., 1]+pSize[..., 1]/2, tCoord[..., 1]+tSize[..., 1]/2)
    right = torch.min(pCoord[..., 0]+pSize[..., 0]/2, tCoord[..., 0]+tSize[..., 0]/2)
    wi = (right-left).clamp(min=0)
    hi = (bottom-top).clamp(min=0)
    intersection = wi*hi
    union = pSize[..., 0]*pSize[..., 1] + tSize[..., 0]*tSize[..., 1] - intersection
    return intersection / (union + 1e-6)

def lossForYOLO(pred, target):
    tCoord = target[..., :2]
    tSize = target[..., 2:4]
    tConf = target[..., 4]

    pCoords = []
    pSizes = []
    pConfs = []
    ious = []
    for box in range(BOXES):
        pCoords.append(torch.sigmoid(pred[..., box*5:box*5+2]))
        pSizes.append(torch.sigmoid(pred[..., box*5+2:(box+1)*5-1]))
        pConfs.append(torch.sigmoid(pred[..., box*5+4]))
        ious.append(intersectionOverUnion(pCoords[box], pSizes[box], tCoord, tSize))

    pCoords = torch.stack(pCoords, dim=-2)
    pSizes = torch.stack(pSizes, dim=-2)
    pConfs = torch.stack(pConfs, dim=-1)
    ious = torch.stack(ious, dim=-1)
    rBox = ious.argmax(dim=-1)
    rBox = rBox.unsqueeze(-1)
    # Nienawidze ksztaltow tensorow
    pCoord = torch.gather(pCoords, dim=-2, index=rBox.unsqueeze(-1).expand(-1, -1, -1, 1, 2)).squeeze(-2)
    pSize = torch.gather(pSizes, dim=-2, index=rBox.unsqueeze(-1).expand(-1, -1, -1, 1, 2)).squeeze(-2)
    pConf = torch.gather(pConfs, dim=-1, index=rBox).squeeze(-1)
    iou = torch.gather(ious, dim=-1, index=rBox).squeeze(-1)


    # Powinny wyjsc gridy SxS wypelnione odpowiednimi danymi

    # Maska pokazujaca w ktorych gridach znajduje sie rzeczywisty obiekt
    objectMask = tConf==1.0
    # Maska pokazujaca w ktorych gridach nie ma obiektow
    noObjectMask = tConf==0.0

    # W pierwszej pracy YOLO loss byl obliczany za pomoca sumy kwadratów roznic, z taka roznica, ze bbox loss byl zwiekszany pieciokrotnie i byl wyliczany jedynie w miejscach gdzie powinny znajdowac sie elementy. Dodatkowo pomniejszamy confidence loss przez 0.5 wtedy kiedy w gridzie nie ma obiektow. Class loss obliczamy normalnie ale tez tylko w miejscach gdzie jest obiekt

    lossBox = 5.0 * nn.MSELoss(reduction="mean")(pCoord[objectMask], tCoord[objectMask])
    lossSize = 5.0* nn.MSELoss(reduction="mean")(torch.sqrt(torch.clamp(pSize[objectMask],min=1e-6)), torch.sqrt(torch.clamp(tSize[objectMask], min=1e-6)))
    lossConfidenceObject = nn.MSELoss(reduction="mean")(pConf[objectMask], iou[objectMask].detach())
    lossConfidenceNoObject = 0.5*nn.MSELoss(reduction="mean")(pConf[noObjectMask],tConf[noObjectMask])
    lossIou = (1 - iou[objectMask].mean())/2

    # SPECJALNIE NA TEN MOMENT POMIJAM LOSS CLASS
    # print(lossBox, lossSize, lossConfidenceObject, lossConfidenceNoObject, lossIou)
    return lossBox + lossSize + lossConfidenceNoObject + lossConfidenceObject + lossIou

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
        target = torch.zeros(GRID, GRID, 5)
        xScale, yScale = (SIZE/w, SIZE/h)
        for ann in self.img_annotations[image_id]:
            x, y, w, h = ann["bbox"]
            # Na tym sie przejechalem ze musi to byc srodek obrazu, a nie np jego lewy gorny rog jak to jest podane w DocLayNet. Siec konwolucyjna widzi srodek obiektu przez jego najwieksze skupienie sie akurat w tym miejscu
            x, y, w, h = (x+w/2)*xScale/SIZE, (y+h/2)*yScale/SIZE, w*xScale/SIZE, h*yScale/SIZE
            gridX, gridY = int(x*GRID), int(y*GRID)
            if target[gridY, gridX, 4]==1.0:
                continue
            x, y = x*GRID - gridX, y*GRID - gridY
            target[gridY, gridX, 0] = x
            target[gridY, gridX, 1] = y
            target[gridY, gridX, 2] = w
            target[gridY, gridX, 3] = h
            target[gridY, gridX, 4] = 1.0
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


class YOLOBasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.cnn = nn.Sequential(
            # Wzor: Out = floor((In + 2Pad - Ker)/Stride) + 1
            #224x224
            ConvBlock(3, 64, 7, 2, 3),
            nn.MaxPool2d((2,2)),
            # 56x56
            ConvBlock(64, 192),
            nn.MaxPool2d((2,2)),
             # 28x28
            ConvBlock(192, 128, 1, 1, 0),
            ConvBlock(128, 256),
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 512),
            nn.MaxPool2d((2,2)),
            # 14x14
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512),
            ConvBlock(512, 512, 1, 1, 0),
            ConvBlock(512, 1024),
            nn.MaxPool2d((2,2)),
            
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024),
            ConvBlock(1024, 1024),

            ConvBlock(1024, 1024),
            ConvBlock(1024, 1024),

        )
        self.head = nn.Conv2d(1024, 5*BOXES, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.head(x)
        # [BATCH, CH, H, W] → [BATCH, H, W, CH]
        return x.permute(0, 2, 3, 1)

