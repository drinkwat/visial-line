# -*- coding: utf-8 -*-
import cv2
import sys
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

cv2.namedWindow("Threshold")
cv2.createTrackbar("TL", "Threshold", 127, 255, nothing)

while True:
    _, img = cap.read()
    
    t_l = cv2.getTrackbarPos("TL", "Threshold")
    
    img = cv2.resize(img, (320, 240))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    _, gray = cv2.threshold(gray, t_l, 255, binary_type)
    
    cv2.imshow('orig', img)
    
    cv2.imshow('Threshold', gray)

    key = cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord(' '):
        cv2.waitKey()
 
cap.release()
cv2.destroyAllWindows()

print(f"\nbinary_l =  {t_l}")
