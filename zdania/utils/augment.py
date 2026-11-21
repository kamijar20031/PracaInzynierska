from PIL import ImageEnhance
import random

def randomBrightness(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    return image

def randomRotate(image):
    image = image.rotate(random.uniform(-3, 3))
    return image

def randomSharpen(image):
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return image