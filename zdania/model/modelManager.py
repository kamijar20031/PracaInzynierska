from utils.modelManager import ModelManager
import torch
from zdania.utils.model import CTCDecode, decodeText, prepareImage
import pickle
from zdania.model.model import OCRModel
from html import escape

class OCRModelManager(ModelManager):
    model = None
    @classmethod
    def init(cls, modelFile, vocabFile):
        char_to_idx = {}
        with open(vocabFile, "rb") as f:
            char_to_idx = pickle.load(f)
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        idx_to_char[0] = "<BLANK>"
        model = OCRModel(len(char_to_idx) + 1)
        model.load_state_dict(torch.load(modelFile, map_location=cls.device))
        model.to(cls.device)
        model.eval()
        cls.decodeDict = idx_to_char
        cls.model = model
        print("Initialized OCR Model")

    @classmethod
    def getTextFromImage(cls, image):
        image = prepareImage(image)
        image = image.to(cls.device)
        with torch.no_grad():
            output = cls.model(image)
        prediction = output.argmax(dim=2)
        prediction = prediction[0].cpu().numpy()
        tokens = CTCDecode(prediction)
        text = escape(decodeText(tokens, cls.decodeDict))
        return text