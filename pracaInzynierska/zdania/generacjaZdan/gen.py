from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import numpy as np
import os 

def render(text):
    fontsize = random.randint(12, 32)
    pad_left = random.randint(0, 25)
    pad_right = random.randint(0, 25)
    pad_top = random.randint(0, 25)
    pad_bottom = random.randint(5, 25)
    fonts = os.listdir("fonts")
    font = ImageFont.truetype(f"fonts/{random.choice(fonts)}", fontsize)
    dummy_img = Image.new("RGB", (1,1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    bbox = dummy_draw.textbbox((0,0), text, font=font)
    width = bbox[2] - bbox[0] + pad_left + pad_right
    height = bbox[3] - bbox[1] + pad_top + pad_bottom
    bg_color = tuple([random.randint(180, 255) for _ in range(3)])
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad_left-bbox[0], pad_top-bbox[1]), text, fill="black", font=font)
    scale = random.uniform(0.8, 1.2)
    new_size = (int(width * scale), int(height * scale))
    img = img.resize(new_size)
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, expand=True, fillcolor=bg_color)
    width, height = img.size
    shear = random.uniform(-0.05, 0.05)
    img = img.transform(
        (width, height),
        Image.AFFINE,
        (1, shear, 0, 0, 1, 0),
        fillcolor=bg_color
    )
    radius=random.uniform(0, 1.2)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    arr = np.array(img)
    noise = np.random.normal(0, 10, arr.shape)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype("uint8")
    img = Image.fromarray(arr)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.5))
    img = img.convert("L")
    return (img, width, height)

with open("wynik.txt","r") as f:
    words = []
    for (i,line) in enumerate(f):
        line = line.strip('\n')
        (img, width, height) =render(line)
        img.save(f"temp/{i}.png")
        try:
            with Image.open(f"temp/{i}.png") as im:
                im.load()
        except:
            os.remove(f"temp/{i}.png")
            print("Obraz do dupy")
        else:
            words.append({"name": line, "file": f"temp/{i}.png"})
    with open("words.txt", 'w') as file:
        for word in words:
            file.write(f"{word['name']} ----- {word['file']}\n")
