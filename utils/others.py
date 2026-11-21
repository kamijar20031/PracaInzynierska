import re

def naturalSort(s):
    a = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower() for text in re.split(a, s)]  