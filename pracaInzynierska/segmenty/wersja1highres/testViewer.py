from torch.utils.data import Dataset, DataLoader
from model import DocLayNetDataset
from PIL import Image, ImageDraw
from consts import *
import torchvision.transforms.functional as functional

train_workers = 10
batch_size = 32
trainDataset = DocLayNetDataset(COCO_TRAIN)
train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=True)

for image, output in train_loader:
    # print(output)
    im = functional.to_pil_image(1 - image[0])
    output = output[0]
    draw = ImageDraw.Draw(im)
    for Y in range(GRID):
        for X in range(GRID):
            confidence = output[Y][X][4]
            bbox = output[Y][X][:4]
            classes = output[Y][X][5:]
            # print(bbox)
            x, y = ((X+bbox[0])*SIZE/GRID, (Y+bbox[1])*SIZE/GRID)
            w, h = (bbox[2]*(SIZE), bbox[3]*(SIZE))
            x -= w/2
            y -= h/2
            draw.line((x, y, x+w, y), fill=(0,0,0,int(255*confidence)))
            draw.line((x+w, y, x+w, y+h), fill=(0,0,0,int(255*confidence)))
            draw.line((x+w, y+h, x, y+h), fill=(0,0,0,int(255*confidence)))
            draw.line((x, y+h, x, y), fill=(0,0,0,int(255*confidence))) 
    im.save(f"results/result-tensor-image.png", "PNG")
    break