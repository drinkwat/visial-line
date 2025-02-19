# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class CameraConfig():
    def __init__(self, name, camera_id, brightness, contrast, exposure, img_size=(320, 240)) -> None:
        self.name = name  # 相机名  Fixed Cam 或者 Mobile Cam
        self.camera_id = camera_id  # 该名对应的相机的设备号 (ls -l /dev/video*  约定 0定 1动)
        self.brightness = brightness  # 亮度
        self.contrast = contrast  # 对比度
        self.exposure = exposure  # 曝光度
        self.img_size = img_size  # 画幅


class PID():
    def __init__(self, kp, ki, kd) -> None:
        self.p = kp  # 比例系数
        self.i = ki  # 积分系数
        self.d = kd  # 微分系数


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=10):
    textColor = textColor[::-1]
    # if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def overlayPNG(imgBack, imgFront, pos=None):
    """
    画面叠加无背景色的png图片
    :param imgBack: 背景图
    :param imgFront: png图
    :param pos: png图左上角点位置
    :return:
    """
    if pos is None:
        pos = [0, 0]

    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack


def eulerAngles2rotationMat(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))


def gate(x, dead_zone=0, activate_zone=0, clamp=0):
    if x > dead_zone:
        return min((x-dead_zone) + activate_zone, clamp)
    elif x < -dead_zone:
        return max((x+dead_zone) - activate_zone, -clamp)
    else:
        return 0
    
    
def my_print(head, head_type='info', content=None):
    if content is None:
        content = ''
    if head_type == 'info':
        bg = 42
        word = 38
    elif head_type == 'warn':
        bg = 43
        word = 31
    elif head_type == 'err':
        bg = 41
        word = 38
    elif head_type == 'data':
        bg = 47
        word = 30
    else:
        bg = 45
        word = 38
    print(f"\n\033[{bg};{word}m   {head}   \033[0m\n{content}\n")
    
