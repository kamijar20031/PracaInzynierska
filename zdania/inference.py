from torchvision import transforms
import torch
from zdania.model.modelManager import OCRModelManager
import os 
from PIL import Image

OCRModelManager.init("model.pt", "vocab.pkl")
imgs = os.listdir("pics")
imgs.sort()
for img in imgs:
    image = Image.open(f"pics/{img}").convert("RGB")
    text = OCRModelManager.getTextFromImage(image)
    print(f"{img} --- {text}")