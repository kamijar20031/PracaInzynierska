from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import numpy as np
import tensorflow as tf

def render(text):
    fontsize = random.randint(12, 32)
    pad_left = random.randint(0, 25)
    pad_right = random.randint(0, 25)
    pad_top = random.randint(0, 25)
    pad_bottom = random.randint(5, 25)
    font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    dummy_img = Image.new("RGB", (1,1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0,0), text, font=font)
    width = bbox[2] - bbox[0] + pad_left + pad_right
    height = bbox[3] - bbox[1] + pad_top + pad_bottom
    bg_color = tuple([random.randint(180, 255) for _ in range(3)])
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad_left-bbox[0], pad_top-bbox[1]), text, fill="black", font=font)
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, expand=True, fillcolor=bg_color)
    shear = random.uniform(-0.2, 0.2)
    img = img.transform(
        (width, height),
        Image.AFFINE,
        (1, shear, 0, 0, 1, 0),
        fillcolor=bg_color
    )
    scale = random.uniform(0.8, 1.2)
    new_size = (int(width * scale), int(height * scale))
    img = img.resize(new_size)
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
    arr = np.array(img)
    return (arr, width, height)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize(image, label, width, height):
    image = image.astype("uint8").tobytes()
    label = label.encode("utf-8")
    feature = {
        "image": _bytes_feature(image),
        "label": _bytes_feature(label),
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

with tf.io.TFRecordWriter("slowa.tfrecord") as writer:
    with open("wynik.txt","r") as f:
        for (i,line) in enumerate(f):
            for j in range(3):
                (img, width, height) =render(line)
                data = serialize(img, line, width, height)
                writer.write(data)
