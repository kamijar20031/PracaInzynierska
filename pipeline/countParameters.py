from modelSegment import UNETBasedModel
from modelOCR import OCRModel
import pickle
import torch

model = UNETBasedModel()
char_to_idx = {}
with open("vocab.pkl", "rb") as f:
    char_to_idx = pickle.load(f)
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
idx_to_char[0] = "<BLANK>"
num_classes = len(char_to_idx) + 1

model2 = OCRModel(num_classes)

def count(model):
    return sum(p.numel() for p in model.parameters())

models = [model, model2]
suma = 0
for i in models:
    print(count(i))