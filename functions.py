import itertools
import operator
import numpy as np
import math
import cv2


def max_by_col(li, col):
    maxx = 0
    try:
        maxx = max(li, key=lambda x: x[col])[col]
    except Exception:
        pass
    return maxx


def min_by_col(li, col):
    minn = 0
    try:
        minn = min(li, key=lambda x: x[col])[col]
    except Exception:
        pass
    return minn


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


def remove_bg(image, bgImg, isBgCaptured):
    if isBgCaptured:
        return cv2.absdiff(bgImg, image)
    return None


def calculate_fingers(res, hull_p, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    fangle = []
    center = None
    if len(hull) > 3:
        # finger tips
        dots = []
        for dot in hull_p:
            dots.append([dot[0][0], dot[0][1]])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        dots = np.float32(dots)
        k = 7
        if len(dots) < k:
            k = len(dots)
        rect, label, center = cv2.kmeans(dots, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        for item in center:
            cv2.circle(drawing, (item[0], item[1]), 8, [211, 84, 200], -1)
        # finger angle
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    fangle.append([far[0], far[1]])
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt, center, fangle
    return False, 0, None, None
