import torch
from PIL import Image, ImageDraw
import cv2
from scipy import ndimage
import torch.nn.functional as F
import os 
from consts import *
import numpy as np
from scipy import ndimage
from collections import defaultdict
from segmenty.UNet.inferenceUtils.masks import Classes, Masks, getPredictionMasks
from segmenty.UNet.model.modelManager import UNetModelManager
from segmenty.UNet.inferenceUtils.others import resizePred
from segmenty.UNet.inferenceUtils.postprocessing import *
from utils.others import naturalSort

UNetModelManager.init("model.pt")

imgs = os.listdir("../../pics")
imgs.sort(key=naturalSort)
for img in imgs[0:60]:
    pred = getPredictionMasks(f"../../pics/{img}")

    colors = np.array([
        [255,255,255],  # BG
        [255,0,0],      # TEXT
        [0,0,255],      # FORMULA
        [128,0,128],    # PICTURE
        [0,255,0],      # TITLE
    ], dtype=np.uint8)

    with Image.open(f"../../pics/{img}").convert("RGBA") as im:
        w, h = im.size
        pred = resizePred((w,h), pred)
        mask = colors[pred]
        mask = Image.fromarray(mask, mode="RGB").convert("RGBA")
        texts, titles, formulas, images = extractElements(pred, (w,h))
        if not os.path.exists(f"results/resultsCrops"):
            os.mkdir(f"results/resultsCrops")

        for i, image in enumerate(images):
            cropped = im.crop(image)
            if not os.path.exists(f"results/resultsCrops/{img}"):
                os.mkdir(f"results/resultsCrops/{img}")
            cropped.save(f"results/resultsCrops/{img}/image-{i}.png", "PNG")
        
        for i, text in enumerate(texts):
            x1,y1,x2,y2 = text["coords"]
            cropped = im.crop((x1,y1,x2,y2))
            if not os.path.exists(f"results/resultsCrops/{img}"):
                os.mkdir(f"results/resultsCrops/{img}")
            cropped.save(f"results/resultsCrops/{img}/text-{i}.png", "PNG")
        
        for i, title in enumerate(titles):
            x1,y1,x2,y2 = title["coords"]
            cropped = im.crop((x1,y1,x2,y2))
            if not os.path.exists(f"results/resultsCrops/{img}"):
                os.mkdir(f"results/resultsCrops/{img}")
            cropped.save(f"results/resultsCrops/{img}/title-{int(title["prob"]*100)}-{i}.png", "PNG")
        
        for i, formula in enumerate(formulas):
            x1,y1,x2,y2 = formula
            cropped = im.crop((x1,y1,x2,y2))
            if not os.path.exists(f"results/resultsCrops/{img}"):
                os.mkdir(f"results/resultsCrops/{img}")
            cropped.save(f"results/resultsCrops/{img}/formula-{i}.png", "PNG")


        overlay = Image.blend(im, mask, alpha=0.4)
        overlay.save(f"results/result-{img}", "PNG")
