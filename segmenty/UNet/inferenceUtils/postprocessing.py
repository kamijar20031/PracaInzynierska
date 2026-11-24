from segmenty.UNet.consts import *
from scipy import ndimage
from segmenty.UNet.inferenceUtils.masks import Classes
import numpy as np
from utils.bboxes import bboxSort, bboxCoverage

def morphBboxes(boxes):
    boxes.sort(key=bboxSort, reverse=True)
    i = 0
    while i < len(boxes):
        biggest = boxes[i]
        group = []
        for box in boxes[i+1:]:
            coverage = bboxCoverage(biggest, box)
            if coverage>=0.8:
                boxes.remove(box)
            if coverage>0 and coverage<1:
                print(f"{i}: {biggest} {box} : {coverage}")
        i+=1
    return boxes

def extractElements(pred, res):
    texts = []
    titles = []
    formulas = []
    images = []
    w, h = res
    elements = (pred!=Classes.BACKGROUND.value)
    label, features = ndimage.label(elements, structure=np.ones((3, 3), dtype=bool))
    objects = ndimage.find_objects(label)
    objects = [(x.start, y.start, x.stop, y.stop) for y,x in objects]
    objects = morphBboxes(objects)
    for region in objects:
        if region is None:
            continue
        x1, y1, x2, y2 = region
        w1 = x2-x1
        h1 = y2-y1
        if w1*h1<600 or h1<5 or w1<5:
            continue
        cnt = np.count_nonzero(pred[y1:y2,x1:x2])
        formulaCnt = np.count_nonzero(pred[y1:y2,x1:x2]==2)
        titleCnt = np.count_nonzero(pred[y1:y2,x1:x2]==4)
        imageCnt = np.count_nonzero(pred[y1:y2,x1:x2]==3)
        data = (x1,y1,x2,y2)
        if imageCnt/cnt>0.3:
            images.append(data)
        elif formulaCnt/cnt > 0.3:
            formulas.append(data)
        elif titleCnt/cnt>0.65:
            titles.append({"coords":data, "prob": titleCnt/cnt})
        else:
            texts.append({"coords":data})
    return texts, titles, formulas, images