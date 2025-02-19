# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

sys.path.append('/home/iflytek/visial_line')
from params import camera_fixed_config, binary_type

def nothing(x):
    pass


cap = cv2.VideoCapture(camera_fixed_config.camera_id)
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

def process_frame(frame, blur_size, low_threshold, high_threshold, kernel_size):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Canny 边缘检测
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # 创建一个闭运算的核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 在闭运算结果上应用dilate和erode操作
    dilated = cv2.dilate(closed, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # 创建掩码，黑色为线条，其它部分为白色
    mask = np.where(eroded == 255, 0, 255).astype(np.uint8)

    return mask





cv2.namedWindow('Processed Frame')
cv2.createTrackbar('GaussianBlur', 'Processed Frame', 3, 21, nothing)
cv2.createTrackbar('Low Threshold', 'Processed Frame', 50, 255, nothing)
cv2.createTrackbar('High Threshold', 'Processed Frame', 150, 255, nothing)
cv2.createTrackbar('Kernel Size', 'Processed Frame', 7, 21, nothing)  # 核大小在7到21之间调整

while True:

    _, img = cap.read()
    img = cv2.resize(img, (320, 240))

    blur_size = cv2.getTrackbarPos('GaussianBlur','Processed Frame')
    low_threshold = cv2.getTrackbarPos('Low Threshold', 'Processed Frame')
    high_threshold = cv2.getTrackbarPos('High Threshold', 'Processed Frame')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Processed Frame')
    
    # 确保kernel_size是奇数，因为形态学操作通常使用奇数大小的核
    kernel_size = max(3, kernel_size // 2 * 2 + 1)
    blur_size = max(3, blur_size // 2 * 2 + 1)

    processed_frame = process_frame(img, blur_size, low_threshold, high_threshold, kernel_size)

    cv2.imshow('orig', img)
    cv2.imshow('Processed Frame', processed_frame)
    
    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()