from zdania.utils.consts import *
from PIL import Image
import torchvision.transforms.functional as functional

def CTCDecode(sequence, blank=0):
    decoded = []
    previous = None
    for token in sequence:
        token = int(token)
        if token == blank:
            previous = token
            continue
        if token == previous:
            continue
        decoded.append(token)
        previous = token
    return decoded

def decodeText(tokens, idx_to_char):
    chars = []
    for token in tokens:
        chars.append(idx_to_char[token])
    return "".join(chars)

def resizeWithPad(image):
    w, h = image.size
    scale = min(width/w, height/h)
    w, h = int(w*scale), int(h*scale)
    imageTemp = image.resize((w,h), Image.Resampling.BILINEAR)
    image = Image.new("RGB", (width, height), (255,255,255))
    x, y = (width - w)/2, (height - h)/2
    image.paste(imageTemp, (int(x), int(y)))
    return image

def prepareImage(image):
    image = resizeWithPad(image)
    image = functional.to_tensor(image)
    image = image.unsqueeze(0)
    return image