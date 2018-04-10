import cv2
import numpy as np
import copy
import math
import time
import csv

# start point/total width
cap_region_x_begin = 0.5
# start point/total width
cap_region_y_end = 0.8
# Binary threshold
threshold = 60
# 1st GaussianBlur parameter
blur_val_1 = 11
# 2nd GaussianBlur parameter
blur_val_2 = 3
# Captured background image
bgImg = np.zeros((480, 640), np.uint8)
# Is background captured
isBgCaptured = False
#
sample = 1
capture = 0
# csvfile
csvfile = open('sample-'+str(sample)+'.csv', 'w', newline='')
csv_wr = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
csv_wr.writerow(['№', 'label', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'a_x1', 'a_y1', 'a_x2', 'a_y2', 'a_x3', 'a_y3', 'a_x4', 'a_y4', 'filename'])

maxdist = 25


def threshold_change(thr):
    print("[INFO] Changed threshold to " + str(thr))


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


def save_data(cap, coords1, coords2, wr, f, img):
    while len(coords1) < 7:
        coords1.append([0, 0])
    while len(coords2) < 4:
        coords2.append([0, 0])

    filename = 'captures/capture-' + str(f) + '-no_' + str(cap) + '.bmp'
    coords = [
        cap,
        f,
        coords1[0][0],
        coords1[0][1],
        coords1[1][0],
        coords1[1][1],
        coords1[2][0],
        coords1[2][1],
        coords1[3][0],
        coords1[3][1],
        coords1[4][0],
        coords1[4][1],
        coords1[5][0],
        coords1[5][1],
        coords1[6][0],
        coords1[6][1],
        coords2[0][0],
        coords2[0][1],
        coords2[1][0],
        coords2[1][1],
        coords2[2][0],
        coords2[2][1],
        coords2[3][0],
        coords2[3][1]
    ]
    coords = [int(x) for x in coords]
    coords.append(filename)
    wr.writerow(coords)
    csvfile.flush()
    cv2.imwrite(filename, img)
    print("[INFO] Row added" + str(cap))


def remove_bg(image):
    global bgImg, isBgCaptured
    if isBgCaptured:
        return cv2.absdiff(bgImg, image)
    return None


def calculate_fingers(res, hull_p, drawing):  # -> finished bool, cnt: finger count
    global maxdist
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
        rect, label, center = cv2.kmeans(dots, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
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


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)
# Binary threshold trackbar
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 255, threshold_change)
# last_time = time.time()
finger_point_pos = None
finger_angle_pos = None

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cropped_img = frame[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

    cv2.imshow('original', frame)

    if isBgCaptured == 1:
        img = remove_bg(cropped_img)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blur_val_1, blur_val_1), 0)

            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

            thresh = cv2.GaussianBlur(thresh, (blur_val_2, blur_val_2), 0)

            # cv2.imshow('blur', blur)
            # cv2.imshow('thresh', thresh)
            cv2.imshow('blur thresh', thresh)

            thresh1 = copy.deepcopy(thresh)
            im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            drawing = np.zeros(img.shape, np.uint8)
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(img.shape, np.uint8)
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                isFinishCal, cnt, finger_point_pos, finger_angle_pos = calculate_fingers(res, hull, drawing)

                if finger_point_pos is not None and finger_angle_pos is not None:
                    min_point_f = [min_by_col(finger_point_pos, 0), min_by_col(finger_point_pos, 1)]
                    min_point_a = [min_by_col(finger_angle_pos, 0), min_by_col(finger_angle_pos, 1)]
                    max_point_f = [max_by_col(finger_point_pos, 0), max_by_col(finger_point_pos, 1)]
                    max_point_a = [max_by_col(finger_angle_pos, 0), max_by_col(finger_angle_pos, 1)]
                    topLeft = [
                        int(min_point_a[0] if min_point_f[0] > min_point_a[0] else min_point_f[0]),
                        int(min_point_a[1] if min_point_f[1] > min_point_a[1] else min_point_f[1])
                    ]
                    bottomRight = [
                        int(max_point_a[0] if max_point_f[0] < max_point_a[0] else max_point_f[0]),
                        int(max_point_a[1] if max_point_f[1] < max_point_a[1] else max_point_f[1])
                    ]
                    hand = cropped_img[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]
                    # cv2.imshow('hand', hand)
                    # print('------------')
                    # print('x: ' + str(max_by_col(finger_point_pos, 0)))
                    # print('y: ' + str(max_by_col(finger_point_pos, 1)))
                    # print('x: ' + str(max_by_col(finger_angle_pos, 0)))
                    # print('y: ' + str(max_by_col(finger_angle_pos, 1)))
                    # print('.............')
                    # print('x: ' + str(min_by_col(finger_point_pos, 0)))
                    # print('y: ' + str(min_by_col(finger_point_pos, 1)))
                    # print('x: ' + str(min_by_col(finger_angle_pos, 0)))
                    # print('y: ' + str(min_by_col(finger_angle_pos, 1)))
                    # print('************')

                    # print('------------')
                    # print(type(finger_point_pos))
                    # print(finger_point_pos)
                    finger_point_pos = list(map(lambda x: [x[0]-min_point_f[0], x[1]-min_point_f[1]], finger_point_pos))
                    # print(finger_point_pos)
                    # print(finger_angle_pos)
                    finger_angle_pos = list(map(lambda x: [x[0] - min_point_a[0], x[1] - min_point_a[1]], finger_angle_pos))
                    # print(finger_angle_pos)
                    # print('************')

                # finger_point_norm = None
                # finger_angle_norm = None
                # finger_point_pos = np.float32(finger_point_pos)
                # finger_angle_pos = np.float32(finger_angle_pos)
                # mina = np.min(finger_angle_pos)
                # minp = np.min(finger_point_pos)
                # print(mina)
                # print(minp)
                # finger_point_norm = cv2.normalize(finger_point_pos, finger_point_norm)
                # finger_angle_norm = cv2.normalize(finger_angle_pos, finger_angle_norm)
                # print('-------------')
                # print(finger_point_pos)
                # print(finger_point_norm)
                # print(finger_angle_pos)
                # print(finger_angle_norm)
                # print('==============')

            cv2.imshow('output', drawing)

    k = cv2.waitKey(5)
    if k == 27 or k == ord('q'):
        csvfile.close()
        break
    elif k == ord('b'):
        bgImg = cropped_img
        isBgCaptured = 1
        print('[INFO] Background Captured')
    elif k == ord('0'):
        save_data(capture, finger_point_pos, finger_angle_pos, csv_wr, 0, thresh)
        capture += 1
    elif k == ord('1'):
        save_data(capture, finger_point_pos, finger_angle_pos, csv_wr, 1, thresh)
        capture += 1
    elif k == ord('2'):
        save_data(capture, finger_point_pos, finger_angle_pos, csv_wr, 2, thresh)
        capture += 1
    elif k == ord('3'):
        save_data(capture, finger_point_pos, finger_angle_pos, csv_wr, 3, thresh)
        capture += 1
    elif k == ord('4'):
        save_data(capture, finger_point_pos, finger_angle_pos, csv_wr, 4, thresh)
        capture += 1
    elif k == ord('5'):
        save_data(capture, finger_point_pos, finger_angle_pos, csv_wr, 5, thresh)
        capture += 1
    elif k == ord('c'):
        save_data(capture, finger_point_pos, finger_angle_pos, csv_wr, 99, thresh)
        capture += 1
        # csv_wr.writerow(
        #     ['№', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'a_x1', 'a_y1',
        #      'a_x2', 'a_y2', 'a_x3', 'a_y3', 'a_x4', 'a_y4'])
        # csv_wr.writerow([
        #     '№',
        #     finger_point_pos[0][0],
        #     finger_point_pos[0][1],
        #     finger_point_pos[1][0],
        #     finger_point_pos[1][1],
        #     finger_point_pos[2][0],
        #     finger_point_pos[2][1],
        #     finger_point_pos[3][0],
        #     finger_point_pos[3][1],
        #     finger_point_pos[4][0],
        #     finger_point_pos[4][1],
        #     finger_point_pos[5][0],
        #     finger_point_pos[5][1],
        #     finger_point_pos[6][0],
        #     finger_point_pos[6][1],
        #     finger_angle_pos[0][0],
        #     finger_angle_pos[0][1],
        #     finger_angle_pos[1][0],
        #     finger_angle_pos[1][1],
        #     finger_angle_pos[2][0],
        #     finger_angle_pos[2][1],
        #     finger_angle_pos[3][0],
        #     finger_angle_pos[3][1]
        # ])
    elif k == ord('n'):
        print('[INFO] Background Captured')
    # print (1 / (time.time() - last_time))
    last_time = time.time()

cv2.destroyAllWindows()
camera.release()
