import os 
from pipeline.consts import *
from segmenty.UNet.inferenceUtils.masks import Classes, Masks, getPredictionMasks
from pipeline.utils.page import Page
from utils.others import naturalSort
from pipeline.utils.docs import *
from segmenty.UNet.model.modelManager import UNetModelManager
from zdania.model.modelManager import OCRModelManager

UNetModelManager.init("modelSegments.pt")
OCRModelManager.init("modelOCR.pt", "vocab.pkl")

pages = []
imgs = os.listdir("../pics")
imgs.sort(key=naturalSort)
for img in imgs[:20]:
    print(f"NOW PROCESSING IMAGE: {img}")
    pred = getPredictionMasks(f"../pics/{img}")
    page = Page()
    page.passData(f"../pics/{img}", pred)
    page.processImage()
    pages.append(page)
    print("\n\n\n")

Document.init(pages)
Document.process()
Document.createEBook()

