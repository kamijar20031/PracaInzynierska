import numpy as np
from enum import Enum
from segmenty.UNet.model.modelManager import UNetModelManager
from collections import defaultdict

class Classes(Enum):
    BACKGROUND = 0
    TEXT = 1
    FORMULA = 2
    IMAGE = 3
    TITLE = 4
    TABLE = 5


class Masks:
    masks = defaultdict(list)

    @staticmethod
    def passPred(pred, predLogits):
        Masks.predTop = pred
        Masks.predLogits = predLogits
        Masks.masks.clear()

    @staticmethod
    def addMaskF(fromMask, toMask, threshold):
        temp = ((Masks.predTop[0] == fromMask.value) & (Masks.predTop[1] == toMask.value) & ((Masks.predLogits[0] - Masks.predLogits[1]) <threshold))
        Masks.masks[toMask.value].append(temp)

    @staticmethod
    def addMaskS(fromMask1, fromMask2, toMask, threshold):
        temp = ((Masks.predTop[0] == fromMask1.value) & (Masks.predTop[1] == fromMask2.value) & (Masks.predTop[2]==toMask.value) & ((Masks.predLogits[0] - Masks.predLogits[2]) <threshold))
        Masks.masks[toMask.value].append(temp)

    @staticmethod
    def getMask(target):
        mask = np.zeros(Masks.predTop.shape[1:], dtype=bool)
        for temp in Masks.masks[target]:
            mask |= temp
        return mask

    @staticmethod
    def getModifiedPredictionMask():
        pred = Masks.predTop[0].copy()
        for target in Masks.masks.keys():
            mask = Masks.getMask(target)
            pred[mask] = target
        return pred



def getPredictionMasks(src):
    predTop, predLogits = UNetModelManager.getTop3ImagePrediction(src)
    Masks.passPred(predTop, predLogits)
    Masks.addMaskF(Classes.TABLE,Classes.BACKGROUND,3)
    Masks.addMaskF(Classes.FORMULA,Classes.BACKGROUND,3)
    Masks.addMaskF(Classes.TEXT,Classes.BACKGROUND,2)
    Masks.addMaskS(Classes.FORMULA,Classes.TEXT,Classes.BACKGROUND,2)
    Masks.addMaskF(Classes.TITLE,Classes.BACKGROUND,2)
    Masks.addMaskS(Classes.TEXT,Classes.TITLE,Classes.BACKGROUND,2)
    Masks.addMaskS(Classes.TEXT,Classes.FORMULA,Classes.BACKGROUND,1.5)
    # Masks.addMaskS(Classes.IMAGE,Classes.TABLE,Classes.BACKGROUND,1.5)
    # Masks.addMaskS(Classes.IMAGE,Classes.TEXT,Classes.BACKGROUND,1.5)
    Masks.addMaskF(Classes.TABLE,Classes.IMAGE,5)
    Masks.addMaskF(Classes.IMAGE,Classes.FORMULA,3)
    Masks.addMaskF(Classes.IMAGE,Classes.TEXT,2)
    
    pred = Masks.getModifiedPredictionMask()

    return pred