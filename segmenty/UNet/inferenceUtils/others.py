import torch.nn.functional as F
import torch
import numpy as np

def resizePred(res, pred):
    w, h = res
    pred = torch.from_numpy(pred.astype(np.int32)).float()
    pred = pred[None][None]
    pred = F.interpolate(pred, size=(h,w), mode="nearest-exact")
    pred = pred[0][0].numpy().astype(np.int32)
    return pred