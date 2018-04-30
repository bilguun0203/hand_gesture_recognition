import cv2
import numpy as np
import copy
import time
from collections import deque
import pyautogui
import data
import functions as func
import save_data as save

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

capture = 0

# neigh = KNeighborsClassifier(n_neighbors=10)
# neigh.fit(tdata, tlabel)


def threshold_change(thr):
    print("[INFO] Changed threshold to " + str(thr))


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)
# Binary threshold trackbar
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 255, threshold_change)
# last_time = time.time()
finger_point_pos = None
finger_angle_pos = None

last_predict = deque([0, 0, 0])
prev = 0

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cropped_img = frame[
                  0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

    cv2.imshow('original', frame)

    if isBgCaptured == 1:
        img = func.remove_bg(cropped_img, bgImg, isBgCaptured)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blur_val_1, blur_val_1), 0)

            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

            thresh = cv2.GaussianBlur(thresh, (blur_val_2, blur_val_2), 0)

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

                isFinishCal, cnt, finger_point_pos, finger_angle_pos = func.calculate_fingers(res, hull, drawing)

                if finger_point_pos is not None and finger_angle_pos is not None:
                    min_point_f = [func.min_by_col(finger_point_pos, 0), func.min_by_col(finger_point_pos, 1)]
                    min_point_a = [func.min_by_col(finger_angle_pos, 0), func.min_by_col(finger_angle_pos, 1)]
                    max_point_f = [func.max_by_col(finger_point_pos, 0), func.max_by_col(finger_point_pos, 1)]
                    max_point_a = [func.max_by_col(finger_angle_pos, 0), func.max_by_col(finger_angle_pos, 1)]
                    topLeft = [
                        int(min_point_a[0] if min_point_f[0] > min_point_a[0] else min_point_f[0]),
                        int(min_point_a[1] if min_point_f[1] > min_point_a[1] else min_point_f[1])
                    ]
                    bottomRight = [
                        int(max_point_a[0] if max_point_f[0] < max_point_a[0] else max_point_f[0]),
                        int(max_point_a[1] if max_point_f[1] < max_point_a[1] else max_point_f[1])
                    ]
                    # hand = cropped_img[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1]]
                    finger_point_pos = list(map(lambda x: [x[0]-min_point_f[0], x[1]-min_point_f[1]], finger_point_pos))
                    finger_angle_pos = list(map(lambda x: [x[0] - min_point_a[0], x[1] - min_point_a[1]], finger_angle_pos))
                    coords1 = finger_point_pos
                    coords2 = finger_angle_pos
                    while len(coords1) < 7:
                        coords1.append([0, 0])
                    while len(coords2) < 4:
                        coords2.append([0, 0])
                    coords = [
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
                    last_predict.popleft()
                    last_predict.append(data.clf.predict([coords]))
                    pred = func.most_common(last_predict)
                    # print(pred)
                    if prev != pred:
                        if prev == 0 and pred == 5:
                            pyautogui.press('space')
                        prev = pred
                        print(pred)
                    # print(clf.predict([coords]))
                    # print(neigh.predict([coords]))
                    # print(neigh.predict_proba([coords]))
            cv2.imshow('output', drawing)

    k = cv2.waitKey(5)
    if k == 27 or k == ord('q'):
        save.csvfile.close()
        break
    elif k == ord('b'):
        bgImg = cropped_img
        isBgCaptured = 1
        print('[INFO] Background Captured')
    elif k == ord('0'):
        save.save_data(capture, finger_point_pos, finger_angle_pos, 0, thresh)
        capture += 1
    elif k == ord('1'):
        save.save_data(capture, finger_point_pos, finger_angle_pos, 1, thresh)
        capture += 1
    elif k == ord('2'):
        save.save_data(capture, finger_point_pos, finger_angle_pos, 2, thresh)
        capture += 1
    elif k == ord('3'):
        save.save_data(capture, finger_point_pos, finger_angle_pos, 3, thresh)
        capture += 1
    elif k == ord('4'):
        save.save_data(capture, finger_point_pos, finger_angle_pos, 4, thresh)
        capture += 1
    elif k == ord('5'):
        save.save_data(capture, finger_point_pos, finger_angle_pos, 5, thresh)
        capture += 1
    elif k == ord('c'):
        save.save_data(capture, finger_point_pos, finger_angle_pos, 99, thresh)
        capture += 1
    elif k == ord('n'):
        print('[INFO] Background Captured')
    # print (1 / (time.time() - last_time))
    last_time = time.time()

cv2.destroyAllWindows()
camera.release()
