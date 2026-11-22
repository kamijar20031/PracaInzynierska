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
    binary = ndimage.binary_dilation(binary, structure=np.ones((1, 5), dtype=bool),iterations=5)
    binary = ndimage.binary_erosion(binary, structure=np.ones((3, 1), dtype=bool),iterations=3)
    binary = ndimage.binary_dilation(binary, structure=np.ones((1, 5), dtype=bool),iterations=5)
    binary = ndimage.binary_dilation(binary, structure=np.ones((1, 3), dtype=bool),iterations=4)
    # img = Image.fromarray(binary)
    # img.save(f"temp/{ran}-dilate.png")
    labels, num_lines = ndimage.label(binary)
    dilated_labels = labels.copy()
    for i in range(1, num_lines + 1):
        line = labels == i
        dilated = ndimage.binary_dilation(line, structure=np.ones((3, 3)), iterations=3)
        # MUSI BYC DILATED_LABELS==0, BO INACZEJ NADPISZE INNE
        dilated_labels[dilated & (dilated_labels == 0)] = i
    regions = ndimage.find_objects(dilated_labels)
    lines = []
    for region in regions:
        if region is None:
            continue
        y1, y2 = region[0].start, region[0].stop
        ys, xs = np.where(binary[y1:y2] > 0)
        if len(xs) == 0:
            continue
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        lines.append((x1, y1, x2, y2))
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
    
    def extractImages(self):
        imagesList, pred = extractImages(self.pred, self.image["res"])
        self.images = imagesList
        self.pred = pred

    def extractTexts(self):
        texts, titles, formulas = extractTexts(self.pred, self.image["res"])
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
                if height>5 and width>5:
                    line = cropped.crop((lx1, ly1, lx2, ly2))
                    line = ImageOps.expand(line, border=int(height/4), fill="white")
                    text = OCRModelManager.getTextFromImage(line)
                    textLines.append(text)
                    # print(f"{text}")
                    # line.save(f"temp/segment-{j}-text-{i}.png")
            segment["text"] = textLines


    def processImage(self):
        self.pred = resizePred(self.image["res"], self.pred)
        self.extractImages()
        self.extractTexts()
        self.predictTextLines()

