# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import cv2AddChineseText

import sys
sys.path.append('/home/iflytek/visial_line')
from params import camera_fixed_config, camera_matrix, distortion_coefficients, projection_matrix, control_pts_offset


marker_id_dict = {0: "左上", 1: "右上", 2: "右下", 3: "左下"}

marker_id = 0  # 标记点编号
state = False  # 是否获取到标记点坐标
temp = None  # 标记点坐标暂存
temp_state = False  # 标记点显示状态 红 暂存态 绿 确认态
marker_camera = np.zeros((4, 2), dtype=np.float32)  # 确认的标记点坐标顺序表


def get_pixel(event, x, y, flags, param):
    global marker_id, state, temp, temp_state, marker_camera

    if not state and event == cv2.EVENT_LBUTTONDOWN:  # 暂未获取到下一控制点坐标 且 鼠标左键按下  记录左键按下点的坐标
        temp = [x, y]  # 暂存坐标
        state = True  # 获取到坐标
        temp_state = False  # 暂存 尚未确认
        print(f"\n当前获得{marker_id_dict.get(marker_id)}的控制点{temp}\n若确认请滚动滚轮，若取消请点击鼠标中键")  # 右键冲突了 ？
    if state and event == cv2.EVENT_MOUSEWHEEL:  # 获取到坐标 且 滚轮确认
        if marker_id & 1:  # 1 3  强制左右高度对齐
            marker_camera[marker_id] = [temp[0], marker_camera[marker_id - 1][1]]  # 记录暂存的坐标
        else:  # 0 2
            marker_camera[marker_id] = temp  # 记录暂存的坐标
        state = False  # 获取完毕 置0 等待下一坐标
        temp_state = True  # 确认态
        print(f"\n{marker_id_dict.get(marker_id)}的控制点{marker_camera[marker_id]}成功记录\n")
        marker_id += 1  # 已记录的控制点+2
    if state and event == cv2.EVENT_MBUTTONDOWN:  # 获取到坐标 且 中键按下
        state = False  # 等待获取下一坐标
        temp = None  # 清空暂存的坐标
        temp_state = False  # 未确认
        print(f"\n已取消当前获取的{marker_id_dict.get(marker_id)}的控制点\n")


cap = cv2.VideoCapture(camera_fixed_config.camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_fixed_config.img_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_fixed_config.img_size[1])
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if w != camera_fixed_config.img_size[0] or h != camera_fixed_config.img_size[1]:
    print(f'Cannot Set to Given Resolution\n'
          f'Currently Set (w, h) to ({w}, {h})')
    camera_fixed_config.img_size = (w, h)

while marker_id != 4:
    ret, img = cap.read()
    # cv2.imshow("test", img)
    img = cv2.flip(img, 1)
    if ret:
        # img = cv2.undistort(src=img, cameraMatrix=camera_matrix, distCoeffs=distortion_coefficients,
        #                     dst=None, newCameraMatrix=projection_matrix)
        img_copy = img.copy()
        # cv2.imshow("test2", img_copy)
        text = f"按空格暂停以记录控制点/继续"
        img = cv2AddChineseText(img, text, (10, 5), (0, 0, 255), 20)

        cv2.imshow(camera_fixed_config.name, img)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord(' '):
            marker_id = 0
            src = img_copy.copy()

            cv2.setMouseCallback(camera_fixed_config.name, get_pixel)

            while 1:
                text = f"当前记录{marker_id_dict.get(marker_id)}的坐标\n若确认请滚动滚轮\n若取消请点击鼠标中键"
                img_copy = cv2AddChineseText(src, text, (10, 5), (0, 0, 255), 20)
                cv2.imshow(camera_fixed_config.name, img_copy)
                if temp is not None and not temp_state:  # 暂存有控制点 且 为暂存态  标红
                    cv2.circle(img_copy, temp, 3, (0, 0, 255), -1)
                    cv2.line(img_copy,
                             (0, int(temp[1])), (camera_fixed_config.img_size[0], int(temp[1])),
                             (0, 0, 255), 1)
                if marker_id:  # 画出记录下的确认的控制点 标绿
                    for i in range(marker_id):
                        cv2.circle(img_copy, marker_camera[i].astype(int), 3, (0, 255, 0), -1)
                        cv2.line(img_copy,
                                 (0, int(marker_camera[i][1])), (camera_fixed_config.img_size[0], int(marker_camera[i][1])),
                                 (255, 255, 0), 1)
                cv2.imshow(camera_fixed_config.name, img_copy)

                if cv2.waitKey(1) & 0xff in [ord(' '), 27]:
                    break
                if marker_id == 4:
                    print('orig\n', marker_camera)
                    delta_upper = marker_camera[1][0] - marker_camera[0][0]
                    delta_lower = marker_camera[2][0] - marker_camera[3][0]
                    marker_camera[0][0] -= control_pts_offset
                    marker_camera[1][0] += control_pts_offset
                    marker_camera[2][0] += round(control_pts_offset * delta_lower / delta_upper)
                    marker_camera[3][0] -= round(control_pts_offset * delta_lower / delta_upper)
                    print('final\n', marker_camera)

                    np.save('perspective.npy', marker_camera)
                    break
    else:
        break

