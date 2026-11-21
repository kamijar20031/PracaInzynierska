import cv2
import numpy as np
import os
from PIL import Image
from scipy import ndimage

def detect_text_lines(crop):
    # crop: obraz PIL zawierający jeden akapit
    gray = np.array(crop.convert("L"))

    # Ciemny tekst -> biały (255), tło -> czarne (0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,  # rozmiar lokalnego sąsiedztwa, musi być nieparzysty
        10   # korekta progu
    )

    # Liczba pikseli tekstu w każdym wierszu
    projection = np.count_nonzero(binary, axis=1)

    # Pomijamy wiersze zawierające jedynie pojedyncze szumy
    active_rows = projection > max(1, int(binary.shape[1] * 0.01))

    # Wykrywanie ciągłych pionowych zakresów
    labels, num_lines = ndimage.label(active_rows)
    regions = ndimage.find_objects(labels)

    lines = []

    for region in regions:
        if region is None:
            continue

        y1, y2 = region[0].start, region[0].stop

        # Dokładny zakres poziomy na podstawie pikseli tekstu
        ys, xs = np.where(binary[y1:y2] > 0)

        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max()) + 1

        lines.append((x1, y1, x2, y2))

    return lines, binary

imgs = os.listdir("lines")
for img in imgs:
    image = Image.open(f"lines/{img}").convert("RGB")
    lines, binary = detect_text_lines(image)

    for i, (lx1, ly1, lx2, ly2) in enumerate(lines):
        line = image.crop((lx1, ly1, lx2, ly2))
        line.save(f"linesResults/{img}-line-{i}.png")