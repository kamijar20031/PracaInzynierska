from PIL import Image, ImageOps
from scipy import ndimage
from zdania.model.modelManager import OCRModelManager
import random
import cv2
from segmenty.UNet.inferenceUtils.others import resizePred
from segmenty.UNet.inferenceUtils.postprocessing import *

def detectTextLines(crop):
    gray = np.array(crop.convert("L"))
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 15)
    # img = Image.fromarray(binary)
    # ran = random.random()
    # img.save(f"temp/{ran}-basic.png")
    binary = ndimage.binary_dilation(binary, structure=np.ones((1, 5), dtype=bool),iterations=2)
    binary = ndimage.binary_erosion(binary, structure=np.ones((3, 1), dtype=bool),iterations=3)
    binary = ndimage.binary_dilation(binary, structure=np.ones((1, 9), dtype=bool),iterations=7)
    binary = ndimage.binary_opening(binary, structure=np.ones((2, 15), dtype=bool))
    blurred = ndimage.gaussian_filter(binary.astype(float), sigma=2)
    binary = blurred > 0.4
    # img = Image.fromarray(binary)
    # img.save(f"temp/{ran}-dilate.png")

    labels, num_lines = ndimage.label(binary)
    dilated_labels = labels.copy()
    
    for i in range(1, num_lines + 1):
        line = labels == i
        pixels = np.count_nonzero(line)
        if pixels<500:
            continue
        dilated = ndimage.binary_dilation(line, structure=np.ones((3, 3)), iterations=5)
        # MUSI BYC DILATED_LABELS==0, BO INACZEJ NADPISZE INNE
        dilated_labels[dilated & (dilated_labels == 0)] = i
    lines = [(x.start,y.start,x.stop,y.stop) for y, x in ndimage.find_objects(dilated_labels)]
    return lines




class Page:

    def __init__(self):
        texts = []
        images = []
        titles = []
        formulas = []
        tables = []
    
    def passData(self, img, predictions):
        self.pred = predictions
        with Image.open(img).convert("RGBA") as im:
            self.image = {"file": img, "res": im.size}
        self.texts = []
        self.titles = []
        self.images = []
        self.formulas = []
        self.tables = []
    
    def extractElements(self):
        texts, titles, formulas, images = extractElements(self.pred, self.image["res"])
        self.images = images
        self.texts = texts
        self.titles = titles
        self.formulas = formulas

    def predictTextLines(self):
        image = Image.open(self.image["file"]).convert("RGB")
        texts = [*self.texts, *self.titles]
        for (j, segment) in enumerate(texts):
            cropped = image.crop(segment["coords"])
            lines = detectTextLines(cropped)
            textLines = []
            for (i, (lx1, ly1, lx2, ly2)) in enumerate(lines):
                height = ly2-ly1
                width = lx2-lx1
                if height>8 and width>5 and height*width>1000:
                    line = cropped.crop((lx1, ly1, lx2, ly2))
                    line = ImageOps.expand(line, border=int(height/4), fill="white")
                    text = OCRModelManager.getTextFromImage(line)
                    textLines.append(text)
                    # print(f"{text}")
                    # line.save(f"temp/segment-{j}-text-{i}.png")
            segment["text"] = textLines


    def processImage(self):
        self.pred = resizePred(self.image["res"], self.pred)
        self.extractElements()
        self.predictTextLines()

