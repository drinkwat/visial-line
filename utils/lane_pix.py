# -*- coding: utf-8 -*-
import cv2
import sys
sys.path.append('/home/iflytek/visial_line')
from params import camera_fixed_config, control_points, dst_pts, camera_matrix, distortion_coefficients, projection_matrix


"""
相机视角显示控制点  用以测量获取 x_real y_real
获取俯视视角下的像素坐标  用以获得 xl_b_pix xr_b_pix
"""


def get_pixel(event, x, y, flags, param):
    global pos

    if event == cv2.EVENT_MOUSEMOVE:  # 暂未获取到下一控制点坐标 且 鼠标左键按下  记录左键按下点的坐标
        pos = [x, y]  # 暂存坐标


pos = [0, 0]

cap = cv2.VideoCapture(camera_fixed_config.camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_fixed_config.img_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_fixed_config.img_size[1])

M = cv2.getPerspectiveTransform(control_points, dst_pts)
cv2.namedWindow(camera_fixed_config.name)
cv2.namedWindow("top down view")
cv2.setMouseCallback("top down view", get_pixel)

while 1:
    re, frame = cap.read()
    if re:
        # frame = cv2.undistort(src=frame, cameraMatrix=camera_matrix, distCoeffs=distortion_coefficients,
        #                       dst=None, newCameraMatrix=projection_matrix)
        cv2.imshow(camera_fixed_config.name, frame)
        for i_p in range(4):
            cv2.circle(frame, control_points[i_p].astype(int), 5, (255, 0, 255), -1)
            cv2.line(frame,
                        control_points[i_p].astype(int),
                        control_points[i_p + 1 if i_p + 1 < 4 else 0].astype(int),
                        (255, 255, 0), 1)
        cv2.imshow(camera_fixed_config.name, frame)

        warped = cv2.warpPerspective(frame, M, camera_fixed_config.img_size, flags=cv2.INTER_LINEAR)
        cv2.circle(warped, pos, 4, (0, 0, 255), -1)
        cv2.putText(warped, f"({pos[0]}, {pos[1]})", (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(0, 0, 255), thickness=2)
        cv2.imshow('top down view', warped)

        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break
        elif k == ord(' '):
            cv2.waitKey(0)
