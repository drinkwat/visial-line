# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
sys.path.append('/home/iflytek/visial_line')
from params import camera_fixed_config


def nothing(x):
    pass


cap = cv2.VideoCapture(camera_fixed_config.camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BRIGHTNESS, camera_fixed_config.brightness)  # 设置亮度
cap.set(cv2.CAP_PROP_CONTRAST, camera_fixed_config.contrast)  # 设置对比度
if camera_fixed_config.exposure:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0~2.6手动曝光  2.6~4自动曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, camera_fixed_config.exposure)
else:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)


print("当前亮度:   ", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("当前对比度: ", cap.get(cv2.CAP_PROP_CONTRAST))
print("当前曝光度: ", cap.get(cv2.CAP_PROP_EXPOSURE))

cv2.namedWindow("HLS")

cv2.createTrackbar("HL", "HLS", 0, 255, nothing)
cv2.createTrackbar("LL", "HLS", 0, 255, nothing)
cv2.createTrackbar("SL", "HLS", 0, 255, nothing)
cv2.createTrackbar("HU", "HLS", 255, 255, nothing)
cv2.createTrackbar("LU", "HLS", 255, 255, nothing)
cv2.createTrackbar("SU", "HLS", 255, 255, nothing)

while True:
    _, img = cap.read()
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    l_h = cv2.getTrackbarPos("HL", "HLS")
    l_s = cv2.getTrackbarPos("LL", "HLS")
    l_v = cv2.getTrackbarPos("SL", "HLS")
    u_h = cv2.getTrackbarPos("HU", "HLS")
    u_s = cv2.getTrackbarPos("LU", "HLS")
    u_v = cv2.getTrackbarPos("SU", "HLS")

    thresh_lower = np.array([l_h, l_s, l_v])
    thresh_upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hls, thresh_lower, thresh_upper)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Orgin", img)
    cv2.imshow("HLS", hls)
    cv2.imshow("Result", result)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"HLS_thresh_lower [{thresh_lower[0]}, {thresh_lower[1]}, {thresh_lower[2]}]")
print(f"HLS_thresh_upper [{thresh_upper[0]}, {thresh_upper[1]}, {thresh_upper[2]}]")
