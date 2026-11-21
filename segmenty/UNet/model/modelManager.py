from segmenty.UNet.model.model import UNETBasedModel
from PIL import Image
import torchvision.transforms.functional as functional
import torch
from segmenty.UNet.consts import *
from utils.modelManager import ModelManager

def prepareImage(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((SIZE, SIZE))
    image = 1.0 - functional.to_tensor(image)
    return image.unsqueeze(0)

# Potem zrob dziedziczenie na ogolnym ModelManager jezeli bedzie potrzebne

class UNetModelManager(ModelManager):
    model = None
    @classmethod
    def init(cls, file):
        model = UNETBasedModel()
        model.to(cls.device)
        model.eval()
        model.load_state_dict(torch.load(file))
        cls.model = model
        print("Initialized UNet Model")

    @classmethod
    def getTop3ImagePrediction(cls, src):
        image = prepareImage(src)
        image = image.to(cls.device)
        with torch.no_grad():
            output = cls.model(image)
        predLogitsT, predTopT = output.topk(3, dim=1)
        predLogits = predLogitsT[0].cpu().numpy()
        predTop = predTopT[0].cpu().numpy()
        return predTop, predLogits