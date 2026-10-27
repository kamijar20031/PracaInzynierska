from model import UNETBasedModel
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as functional
import torch.nn.functional as F
import os 
from consts import *
import numpy as np
from scipy import ndimage
from enum import Enum
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETBasedModel()
model.to(device)
model.eval()

model.load_state_dict(torch.load("model.pt"))

class Classes(Enum):
    BACKGROUND = 0
    TEXT = 1
    FORMULA = 2
    IMAGE = 3
    TITLE = 4
    TABLE = 5


class Masks:
    masks = defaultdict(list)

    @staticmethod
    def passPred(pred, predLogits):
        Masks.predTop = pred
        Masks.predLogits = predLogits
        Masks.masks.clear()

    @staticmethod
    def addMaskF(fromMask, toMask, threshold):
        temp = ((Masks.predTop[0] == fromMask.value) & (Masks.predTop[1] == toMask.value) & ((Masks.predLogits[0] - Masks.predLogits[1]) <threshold))
        Masks.masks[toMask.value].append(temp)

    @staticmethod
    def addMaskS(fromMask1, fromMask2, toMask, threshold):
        temp = ((Masks.predTop[0] == fromMask1.value) & (Masks.predTop[1] == fromMask2.value) & (Masks.predTop[2]==toMask.value) & ((Masks.predLogits[0] - Masks.predLogits[2]) <threshold))
        Masks.masks[toMask.value].append(temp)

    @staticmethod
    def getMask(target):
        mask = np.zeros(Masks.predTop.shape[1:], dtype=bool)
        for temp in Masks.masks[target]:
            mask |= temp
        return mask

    @staticmethod
    def getModifiedPredictionMask():
        pred = Masks.predTop[0].copy()
        for target in Masks.masks.keys():
            mask = Masks.getMask(target)
            pred[mask] = target
        return pred



def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((SIZE, SIZE))
    image = 1.0 - functional.to_tensor(image)
    return image.unsqueeze(0)

imgs = os.listdir("../../pics")
for img in imgs:
    image = prepare_image(f"../../pics/{img}")
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    predLogitsT, predTopT = output.topk(3, dim=1)
    predLogits = predLogitsT[0].cpu().numpy()
    predTop = predTopT[0].cpu().numpy()

    Masks.passPred(predTop, predLogits)
    Masks.addMaskF(Classes.TABLE,Classes.BACKGROUND,3)
    Masks.addMaskF(Classes.FORMULA,Classes.BACKGROUND,3)
    Masks.addMaskF(Classes.TEXT,Classes.BACKGROUND,2)
    Masks.addMaskS(Classes.FORMULA,Classes.TEXT,Classes.BACKGROUND,2)
    Masks.addMaskF(Classes.TITLE,Classes.BACKGROUND,2)
    Masks.addMaskS(Classes.TEXT,Classes.TITLE,Classes.BACKGROUND,2)
    Masks.addMaskS(Classes.TEXT,Classes.FORMULA,Classes.BACKGROUND,1.5)
    # Masks.addMaskS(Classes.IMAGE,Classes.TABLE,Classes.BACKGROUND,1.5)
    # Masks.addMaskS(Classes.IMAGE,Classes.TEXT,Classes.BACKGROUND,1.5)
    Masks.addMaskF(Classes.TABLE,Classes.IMAGE,5)
    Masks.addMaskF(Classes.IMAGE,Classes.FORMULA,3)
    Masks.addMaskF(Classes.IMAGE,Classes.TEXT,2)
    
    pred = Masks.getModifiedPredictionMask()

    colors = np.array([
        [255,255,255],  # BG
        [255,0,0],      # TEXT
        [0,0,255],      # FORMULA
        [128,0,128],    # PICTURE
        [0,255,0],      # TITLE
        [20,20,20],     # TABLE
    ], dtype=np.uint8)
    
    with Image.open(f"../../pics/{img}").convert("RGBA") as im:
        w, h = im.size

        pred = torch.from_numpy(pred.astype(np.int32)).float()
        pred = pred[None][None]
        pred = F.interpolate(pred, size=(h,w), mode="nearest-exact")
        pred = pred[0][0].numpy().astype(np.int32)
        mask = colors[pred]

        mask = Image.fromarray(mask, mode="RGB").convert("RGBA")
        imgs = (pred==Classes.IMAGE.value)
        imgsDil = ndimage.binary_dilation(imgs, structure=np.ones((3, 3), dtype=bool),iterations=7)
        label, features = ndimage.label(imgsDil, structure=np.ones((3, 3), dtype=bool))
        objects = ndimage.find_objects(label)
        for i, region in enumerate(objects):
            if region is None:
                continue
            y, x = region
            x1, x2 = x.start, x.stop
            y1, y2 = y.start, y.stop
            scaledArea = (x2-x1)*(y2-y1)*(SIZE/w)*(SIZE/h)
            if scaledArea<1000:
                continue
            cropped = im.crop((x1,y1,x2,y2))
            if not os.path.exists(f"../results/resultsCrops/{img}"):
                os.mkdir(f"../results/resultsCrops/{img}")
            cropped.save(f"../results/resultsCrops/{img}/image-{i}.png", "PNG")
            offsetX = int((x2-x1)*0.1)
            offsetY = int((y2-y1)*0.1)
            pred[y1:y2, x1:x2] = Classes.BACKGROUND.value
        text = (pred!=Classes.BACKGROUND.value)
        label, features = ndimage.label(text, structure=np.ones((3, 3), dtype=bool))
        objects = ndimage.find_objects(label)
        for i, region in enumerate(objects):
            if region is None:
                continue
            y, x = region
            x1, x2 = x.start, x.stop
            y1, y2 = y.start, y.stop
            if (x2-x1)*(y2-y1)<600:
                continue
            cropped = im.crop((x1,y1,x2,y2))
            if not os.path.exists(f"../results/resultsCrops/{img}"):
                os.mkdir(f"../results/resultsCrops/{img}")
            cnt = np.count_nonzero(pred[y1:y2,x1:x2])
            formulaCnt = np.count_nonzero(pred[y1:y2,x1:x2]==2)
            titleCnt = np.count_nonzero(pred[y1:y2,x1:x2]==4)
            typ = "text"
            if formulaCnt/cnt > 0.3:
                typ = "formula"
            elif titleCnt/cnt>0.5:
                typ="title"
            cropped.save(f"../results/resultsCrops/{img}/{typ}-{i}.png", "PNG")
        print(im.size, mask.size)
        overlay = Image.blend(im, mask, alpha=0.4)
        overlay.save(f"../results/result-{img}", "PNG")
