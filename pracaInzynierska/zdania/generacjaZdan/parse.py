import xml.etree.ElementTree as ET
import mwparserfromhell
import re
import random
def clean_text(text):
    text = mwparserfromhell.parse(text).strip_code()
    text = re.sub(r"Kategoria:.*", "", text)
    text = re.sub(r"(thumb\|.*)", "", text)
    text = re.sub(r"(Plik:.*)", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"==.*?==", "", text)
    text = re.sub(r"^[\*\-].*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_wiki(path):
    context = ET.iterparse(path, events=("end",))
    for event, elem in context:
        if elem.tag.endswith("page"):
            text_elem = elem.find(".//{*}text")
            text = text_elem.text if text_elem is not None else ""
            text = clean_text(text)
            yield text
            elem.clear()

with open("wynik.txt", "w") as f:
    for i, text in enumerate(parse_wiki("plwiki.xml")):
        words = text.split()
        sentences = []
        j = 0
        while j < len(words):
            size = random.randint(1,10)
            sentences.append(" ".join(words[j:j+size]))
            j+=size
        for sentence in sentences:
            if sentence!=" " and sentence!="":
                while sentence[-1] ==" ":
                    sentence = sentence[:-1]
                while sentence[0] == " ":
                    sentence = sentence[1:]
                f.write(f"{sentence}\n")
        if i>200:
            break;
