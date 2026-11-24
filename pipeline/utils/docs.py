import uuid
from io import BytesIO
from ebooklib import epub
from PIL import Image
import cv2
import numpy as np
from pipeline.utils.XYCut import *

def createImageData(data, filename):
    buffer= BytesIO()
    data.save(buffer, format="PNG")
    image = epub.EpubImage(uid=f"Obraz-{filename}", file_name=f"Images/{filename}", media_type="image/png", content=buffer.getvalue())
    Document.addImage(image)

def addResponsiveImageVersions(data, sizes):
    variants = []
    w,h = data.size
    for width in sizes:
        if w<width:
            break
        copy = data.copy()
        copy.thumbnail((width, h), Image.Resampling.LANCZOS)
        variants.append(copy)
    return variants

class DocPart:
    def __init__(self, data, coords):
        self.data = data
        self.coords = coords
    def __str__(self):
        return f"Coords: {self.coords}"


class DocImage(DocPart):
    num=0
    def __init__(self, data, coords):
        super().__init__(data, coords)
        self.type="image"
    def __repr__(self):
        return f"Image: {super().__str__()}"

    def addToPage(self):
        DocImage.num+=1
        sizes = [512,768,1024]
        w = self.data.size[0]
        variants = addResponsiveImageVersions(self.data, sizes)
        base = f"image-{uuid.uuid4().hex}"
        for i, variant in enumerate(variants):
            name = f"{base}-{sizes[i]}.png"
            createImageData(variant, name)
        smallest = min(w, sizes[0])
        if smallest==w:
            name = f"{base}-{w}.png"
            createImageData(self.data, name)
        srcset = ""
        if len(variants)!=0:
            srcset = ", ".join([f"../Images/{base}-{width}.png {width}w" for width in sizes[:len(variants)]])
            srcset = f"srcset='{srcset}'"
        
        return f"<img src='../Images/{base}-{smallest}.png'  alt='Obraz {DocImage.num}' {srcset} sizes='100vw'/>"


class DocFormula(DocPart):
    num=1
    def __init__(self, data, coords):
        super().__init__(data, coords)
        self.type="formula"
    def __repr__(self):
        return f"Image: {super().__str__()}"

    def addToPage(self):
        DocFormula.num+=1
        sizes = [512,768,1024]
        w = self.data.size[0]
        variants = addResponsiveImageVersions(self.data, sizes)
        base = f"formula-{uuid.uuid4().hex}"
        for i, variant in enumerate(variants):
            name = f"{base}-{sizes[i]}.png"
            createImageData(variant, name)
        smallest = min(w, sizes[0])
        if smallest==w:
            name = f"{base}-{w}.png"
            createImageData(self.data, name)
        srcset = ""
        if len(variants)!=0:
            srcset = ", ".join([f"../Images/{base}-{width}.png {width}w" for width in sizes[:len(variants)]])
            srcset = f"srcset='{srcset}'"
        return f"<img src='../Images/{base}-{smallest}.png'  alt='Wzór {DocFormula.num}' {srcset} sizes='100vw'/>"




class DocText(DocPart):
    def __init__(self, data, coords):
        super().__init__(data, coords)
        self.type="text"
    def __repr__(self):
        return f"Text: {super().__str__()}"

    def addToPage(self):
        return f"<p>{self.data}</p>"

class DocTitle(DocPart):
    def __init__(self, data, coords, prob):
        super().__init__(data, coords)
        self.type="title"
        self.prob=prob
    def __repr__(self):
        return f"Title: {super().__str__()}"

    def addToPage(self):
        if self.prob>0.8:
            return f"<h1>{self.data}</h1>"
        else:
            return f"<h2>{self.data}</h2>"

class Document():
    @classmethod
    def init(cls, pages):
        cls.pages = pages
        cls.parts = []
        cls.book = epub.EpubBook()

    @classmethod
    def addImage(cls, image):
        cls.book.add_item(image)

    @classmethod
    def process(cls):
        for page in cls.pages:
            elements = []
            # Moze pozniej mozna dodac jakas bardziej skomplikowana logike łączenia linijek
            for segment in page.texts:
                text = " ".join(segment["text"])
                elements.append(DocText(text, segment["coords"]))
            for segment in page.titles:
                text = " ".join(segment["text"])
                elements.append(DocTitle(text, segment["coords"], segment["prob"]))
            img = Image.open(page.image["file"]).convert("RGB")
            for image in page.images:
                crop = img.crop(image)
                elements.append(DocImage(crop, image))
            for formula in page.formulas:
                crop = img.crop(formula)
                elements.append(DocFormula(crop, formula))
            w, h = img.size
            
            temp = elements
            elements = XYCut(elements, w, h,  w/100)
            if len(elements)!=len(temp):
                print(page.image["file"])
                print(len(temp), len(elements))
            for el in elements:
                cls.parts.append(el.addToPage())
    
    @classmethod
    def createEBook(cls):
        
        # Moze pozniej jakis podzial na rozdzialy po tytulach? Byloby bardzo trudno.
        chapter = epub.EpubHtml(title="Test", file_name="Text/chapter1.xhtml")
        chapter.content = "".join(cls.parts)
        cls.book.add_item(chapter)
        cls.book.toc = (chapter,)
        cls.book.add_item(epub.EpubNcx())
        cls.book.add_item(epub.EpubNav())
        cls.book.spine = ["nav", chapter]
        epub.write_epub("test.epub", cls.book)
