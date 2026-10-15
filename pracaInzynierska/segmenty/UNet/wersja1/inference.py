from model import UNETBasedModel
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as functional
import os 
from consts import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETBasedModel()
model.to(device)
model.eval()

model.load_state_dict(torch.load("model.pt"))

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
    pred = output.argmax(dim=1)[0].cpu().numpy()
    colors = np.array([
        [255,255,255],  # BG
        [0,255,0],      # CAPTION
        [255,255,0],    # FOOTNOTE
        [0,255,255],    # FORMULA
        [200,0,100],      # LIST ITEM
        [200,100,0],    # PAGE FOOTER
        [0,128,128],    # PAGE HEADER
        [128,0,128],    # PICTURE
        [128,128,0],    # SECTION HEADER
        [128,128,128],  # TABLE
        [255,0,0],      # TEXT
        [255,0,255],    # TITLE
    ], dtype=np.uint8)
    mask = colors[pred]
    mask = Image.fromarray(mask, mode="RGB").convert("RGBA")
    with Image.open(f"../../pics/{img}").convert("RGBA") as im:
        w, h = im.size
        im = im.resize((SIZE,SIZE))
        print(mask.size, im.size)
        overlay = Image.blend(im, mask, alpha=0.4)
        overlay.save(f"../results/result-{img}", "PNG")
