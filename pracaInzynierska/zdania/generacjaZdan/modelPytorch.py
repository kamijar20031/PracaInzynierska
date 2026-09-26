import torch
import torch.nn as nn
import editdistance
from PIL import Image, ImageEnhance
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

DATASET_NAME = "temp"
height = 32
width = 512
train_epochs=100
learning_rate = 0.0005
patience = 20
split = 0.9
batch_size = 16
train_workers = 10

# Augmentacja danych

def randomBrightness(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    return image

def randomRotate(image):
    image = image.rotate(random.uniform(-3, 3))
    return image

def randomSharpen(image):
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return image


# CTC

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

# CER

def calculate_cer(outputs, labels, idx_to_char, blank=0):
    predictions = outputs.argmax(dim=2)
    total_distance = 0
    total_chars = 0
    for pred, target in zip(predictions, labels):
        pred = pred.cpu().numpy()
        pred_tokens = ctc_decode(pred, blank)
        pred_text = decode_text(pred_tokens, idx_to_char)
        target_tokens = target.tolist()
        target_text = decode_text(target_tokens, idx_to_char)
        total_distance += editdistance.eval(pred_text, target_text)
        total_chars += len(target_text)
    return total_distance / max(total_chars,1)

# Ładowanie danych 
class OCRDataset(Dataset):

    def __init__(self, DATASET_NAME, target_width, target_height, mytransforms=None):
        self._transforms = mytransforms or []
        self._name = DATASET_NAME
        self._t_w = target_width
        self._t_h = target_height
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
        words = open(os.path.join(self._name,"words.txt"), "r").readlines()
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

    def _resize_with_pad(self, image):
        w, h = image.size
        scale = min(self._t_w/w, self._t_h/h)
        w, h = int(w*scale), int(h*scale)
        imageTemp = image.resize((w,h), Image.Resampling.BILINEAR)
        image = Image.new("RGB", (self._t_w, self._t_h), (255,255,255))
        x, y = (self._t_w - w)/2, (self._t_h - h)/2
        image.paste(imageTemp, (int(x), int(y)))
        return image

    def split(self, rate):
        train_size = int(rate * self.__len__())
        val_size = self.__len__() - train_size
        return random_split(self, [train_size, val_size])

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        label = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        image = Image.open(path).convert("RGB")
        image = self._resize_with_pad(image)
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
            # 512x32
            ConvBlock(3,32),
            # 512 x 32
            nn.MaxPool2d((2,2)),
            # 256 x 16
            ConvBlock(32,64),
            # 256 x 16
            ConvBlock(64,128),
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

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor(
        [len(label) for label in labels],
        dtype=torch.long
    )
    labelsConcat = torch.cat(labels)
    return images, labelsConcat, label_lengths, labels

