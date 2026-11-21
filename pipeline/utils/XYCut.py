import numpy as np

def XYCut(e, width, height, gap):
    def getGaps(proj, offset):
        if proj[0]==0 or proj[-1]==0:
            indices = np.where(proj==1)[0]
            if proj[0]==0:
                proj[:indices[0]+1] = 1
            if proj[-1]==0:
                proj[indices[-1]-1:] = 1
        # Bardzo sprytny pomysł na znalezienie krawędzi
        edges = np.diff(np.pad(proj, (1,1), constant_values=(1,1)))
        starts = np.where(edges==-1)[0]
        ends = np.where(edges==1)[0]
        return [(start+offset, end+offset) for start, end in zip(starts, ends) if end-start >=gap]

    results = []

    def separateElements(elements, gaps, vertical, start, end):
        gaps.append(end)
        separated = [[] for _ in range(len(gaps))]
        separatedStarts = []
        separatedEnds = []
        for element in elements:
            for i, middle in enumerate(gaps):
                if element.coords[2+vertical] < middle:
                    separated[i].append(element)
                    break
        prevStart = start
        for middle in gaps:
            separatedStarts.append(prevStart)
            separatedEnds.append(middle)
            prevStart += middle
        return separated, separatedStarts, separatedEnds

    def getBboxProjection(elements, start, end, vertical):
        size = end - start
        proj = np.zeros(size)
        for element in elements:
            relStart = element.coords[vertical] - start
            relEnd = element.coords[2+vertical] - start
            proj[relStart:relEnd] = 1
        return proj

    def r(elements, x0, y0, x1, y1, vertical, start=False):
        val1 = x0
        val2 = x1
        if vertical:
            val1 = y0
            val2 = y1
        w = x1-x0
        h = y1-y0
        if len(elements)==1:
            if not start:
                results.extend(elements)
            return False
        if not elements or w<30 or h<30:
            return
        proj = getBboxProjection(elements, val1, val2, vertical)
        gaps = getGaps(proj, val1)
        if gaps:
            mids = [(end+start)//2 for (start, end) in gaps]
            separated, starts, ends = separateElements(elements, mids, vertical, val1, val2)
            for i, group in enumerate(separated):
                if vertical:
                    r(group, x0, starts[i], x1, ends[i], False)
                else:
                    r(group, starts[i], y0, ends[i], y1, True)
            return True
        else:
            # Dojdzie do takiej sytuacji gdy np jest zbyt male rozgraniczenie
            if not start:
                elements.sort(key=lambda e: e.coords[2+vertical])
                results.extend(elements)
            return False
    if not r(e, 0, 0, width, height, False, True):
        r(e, 0, 0,  width, height, True)
    return results