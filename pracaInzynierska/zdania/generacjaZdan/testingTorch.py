from modelPytorch import OCRModel, ctc_decode, decode_text
from torchvision import transforms
import torch
import pickle
from PIL import Image, ImageEnhance
import os 

height = 32
width = 512

def _resize_with_pad(image):
    w, h = image.size
    scale = min(width/w, height/h)
    w, h = int(w*scale), int(h*scale)
    imageTemp = image.resize((w,h), Image.Resampling.BILINEAR)
    image = Image.new("RGB", (width, height), (255,255,255))
    x, y = (width - w)/2, (height - h)/2
    image.paste(imageTemp, (int(x), int(y)))
    return image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_to_idx = {}

with open("vocab.pkl", "rb") as f:
    char_to_idx = pickle.load(f)

idx_to_char = {idx: char for char, idx in char_to_idx.items()}
idx_to_char[0] = "<BLANK>"
num_classes = len(char_to_idx) + 1
model = OCRModel(num_classes)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

test_transform = transforms.Compose([transforms.ToTensor()])

def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image = _resize_with_pad(image)
    image = test_transform(image)
    image = image.unsqueeze(0)
    return image

imgs = os.listdir("pics")
for img in imgs:
    image = prepare_image(f"pics/{img}")
    image = image.to(device)
    with torch.no_grad():
        output = model(image)

    prediction = output.argmax(dim=2)
    prediction = prediction[0].cpu().numpy()
    tokens = ctc_decode(prediction)
    text = decode_text(tokens, idx_to_char)
    print(text)