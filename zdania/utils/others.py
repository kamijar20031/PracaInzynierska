import torch
from zdania.utils.model import *
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor(
        [len(label) for label in labels],
        dtype=torch.long
    )
    labelsConcat = torch.cat(labels)
    return images, labelsConcat, label_lengths, labels

def calculateCer(outputs, labels, idx_to_char, blank=0):
    predictions = outputs.argmax(dim=2)
    total_distance = 0
    total_chars = 0
    for pred, target in zip(predictions, labels):
        pred = pred.cpu().numpy()
        pred_tokens = CTCDecode(pred, blank)
        pred_text = decodeText(pred_tokens, idx_to_char)
        target_tokens = target.tolist()
        target_text = decodeText(target_tokens, idx_to_char)
        total_distance += editdistance.eval(pred_text, target_text)
        total_chars += len(target_text)
    return total_distance / max(total_chars,1)