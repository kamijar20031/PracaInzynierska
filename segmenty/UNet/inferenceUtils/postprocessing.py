from segmenty.UNet.consts import *
from scipy import ndimage
from segmenty.UNet.inferenceUtils.masks import Classes
import numpy as np

def extractImages(pred, res):
    imagesList = []
    w, h = res
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
        w1 = x2-x1
        h1 = y2-y1
        scaledArea = w1*h1*(SIZE/w)*(SIZE/h)
        if scaledArea<1000 or w1<30 or h1<30:
            continue
        imagesList.append((x1,y1,x2,y2))
        pred[y1:y2, x1:x2] = Classes.BACKGROUND.value
    return imagesList, pred

def extractTexts(pred, res):
    texts = []
    titles = []
    formulas = []
    w, h = res
    text = (pred!=Classes.BACKGROUND.value)
    label, features = ndimage.label(text, structure=np.ones((3, 3), dtype=bool))
    objects = ndimage.find_objects(label)
    for region in objects:
        if region is None:
            continue
        y, x = region
        x1, x2 = x.start, x.stop
        y1, y2 = y.start, y.stop
        w1 = x2-x1
        h1 = y2-y1
        if w1*h1<600 or h1<5 or w1<5:
            continue
        cnt = np.count_nonzero(pred[y1:y2,x1:x2])
        formulaCnt = np.count_nonzero(pred[y1:y2,x1:x2]==2)
        titleCnt = np.count_nonzero(pred[y1:y2,x1:x2]==4)
        data = (x1,y1,x2,y2)
        if formulaCnt/cnt > 0.3:
            formulas.append(data)
        elif titleCnt/cnt>0.65:
            titles.append({"coords":data, "prob": titleCnt/cnt})
        else:
            texts.append({"coords":data})
    return texts, titles, formulas