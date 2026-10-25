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
    predLogits, predTop = output.topk(3, dim=1)
    predLogits = predLogits[0].cpu().numpy()
    predTop = predTop[0].cpu().numpy()

    table_to_bg = (
        ((predTop[0] == 5) | (predTop[0] == 2)) & (predTop[1]==0) & ((predLogits[0] - predLogits[1]) <3)
    )
    formula_to_bg = (
        (predTop[0] == 2) & (predTop[1]==0) & ((predLogits[0] - predLogits[1]) <1)
    )
    text_to_bg = (
        (predTop[0] == 1) & (predTop[1]==0) & ((predLogits[0] - predLogits[1]) <2)
    )
    formula_text_to_bg = (
        (predTop[0] == 2) & (predTop[1] == 1) & (predTop[2]==0) & ((predLogits[0] - predLogits[2]) <1.5)
    )
    title_to_bg = (
        (predTop[0] == 4) & (predTop[1]==0) & ((predLogits[0] - predLogits[1]) <2)
    )
    text_title_to_bg = (
        (predTop[0] == 1) & (predTop[1] == 4) & (predTop[2]==0) & ((predLogits[0] - predLogits[2]) <2)
    )
    text_formula_to_bg = (
        (predTop[0] == 1) & (predTop[1] == 2) & (predTop[2]==0) & ((predLogits[0] - predLogits[2]) <1.5)
    )
    bg = (table_to_bg | formula_to_bg | text_to_bg | formula_text_to_bg | text_formula_to_bg | title_to_bg | text_title_to_bg)

    table_to_img = (
        (predTop[0] == 5) & (predTop[1] == 3) & ((predLogits[0] - predLogits[1]) < 5)
    )

    img_to_formula = (
        (predTop[0] == 3) & (predTop[1] == 2) & ((predLogits[0] - predLogits[1]) < 3)
    )

    pred = predTop[0]
    pred[bg] = 0
    pred[table_to_img] = 3
    pred[img_to_formula] = 2
    colors = np.array([
        [255,255,255],  # BG
        [255,0,0],      # TEXT
        [0,0,255],      # FORMULA
        [128,0,128],    # PICTURE
        [0,255,0],      # TITLE
        [20,20,20],     # TABLE
    ], dtype=np.uint8)
    mask = colors[pred]
    mask = Image.fromarray(mask, mode="RGB").convert("RGBA")
    with Image.open(f"../../pics/{img}").convert("RGBA") as im:
        w, h = im.size
        im = im.resize((SIZE,SIZE))
        print(mask.size, im.size)
        overlay = Image.blend(im, mask, alpha=0.4)
        overlay.save(f"../results/result-{img}", "PNG")
