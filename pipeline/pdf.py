import pymupdf

doc = pymupdf.open("../pdf/bazy.pdf")
pages = doc[0:]
for (i, page) in enumerate(pages):
    pix = page.get_pixmap(dpi=150)
    print(pix.width, pix.height)
    pix.save(f"../pics/{i}.png")