
def bboxCoverage(bbox1, bbox2):
    # bbox1 jest zawsze wiekszy niz bbox2
    xa0, ya0, xa1, ya1 = bbox1
    xb0, yb0, xb1, yb1 = bbox2
    wb, hb = xb1 - xb0, yb1 - yb0
    x0 = max(xa0, xb0)
    x1 = min(xa1, xb1)
    y0 = max(ya0, yb0)
    y1 = min(ya1, yb1)
    w, h = max(x1 - x0, 0), max(y1 - y0, 0)
    if w*h==0:
        return 0
    return min(1, (w*h)/(wb*hb))

def bboxSort(bbox):
    x0,y0,x1,y1 = bbox
    return (x1-x0)*(y1-y0)