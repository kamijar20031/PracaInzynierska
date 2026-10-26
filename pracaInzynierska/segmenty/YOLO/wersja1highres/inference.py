from model import YOLOBasedModel, intersectionOverUnion
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as functional
import os 
from consts import *

threshold = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLOBasedModel()
model.to(device)
model.eval()

model.load_state_dict(torch.load("model.pt"))

def prepare_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((SIZE, SIZE))
    image = 1.0 - functional.to_tensor(image)
    return image.unsqueeze(0)

imgs = os.listdir("../../pics")
for img in imgs:
    image = prepare_image(f"../../pics/{img}")
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    print(output.shape)
    output = output[0]
    boxes = []
    with Image.open(f"../../pics/{img}").convert("RGBA") as im:
        w, h = im.size
        xScale, yScale = (w/SIZE, h/SIZE)
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for Y in range(GRID):
            for X in range(GRID):
                for box in range(BOXES):
                    confidence = (torch.sigmoid(output[Y][X][box*5 + 4]))
                    coord = (torch.sigmoid(output[Y][X][box*5:2+box*5]))
                    size = (torch.sigmoid(output[Y][X][box*5+2:4+box*5]))
                    x, y = ((X+coord[0])*SIZE/GRID*xScale, (Y+coord[1])*SIZE/GRID*yScale)
                    w, h = (size[0]*SIZE*xScale, size[1]*SIZE*yScale)
                    boxes.append({"p": confidence, "c": (x,y), "s": (w,h)})
        boxes = list(filter(lambda box: box["p"]>0.3, boxes))
        boxes.sort(key=lambda box: box["p"], reverse=True)
        i = 0
        while i < len(boxes):
            best = boxes[i]
            for box in boxes[i+1:]:
                iou = intersectionOverUnion(torch.stack(best["c"]),torch.stack(best["s"]),
                                            torch.stack(box["c"]), torch.stack(box["s"])).item()
                if iou >= 0.4:
                    boxes.remove(box)
            i+=1

        for box in boxes:
            confidence = box["p"]
            w, h = box["s"]
            x, y = box["c"]
            x -= w/2
            y -= h/2
            gray = int(255 * confidence.item())
            draw.line((x, y, x+w, y), fill=(0,0,0,gray))
            draw.line((x+w, y, x+w, y+h), fill=(0,0,0,gray))
            draw.line((x+w, y+h, x, y+h), fill=(0,0,0,gray))
            draw.line((x, y+h, x, y), fill=(0,0,0,gray))

        im = Image.alpha_composite(im, overlay)
        im.save(f"../results/result-{img}", "PNG")
