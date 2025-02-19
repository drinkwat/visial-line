# -*- coding: utf-8 -*-
import cv2
import numpy as np
try:
    from params import binary_type, binary_l
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/jetson/2024')
    from params import binary_type, binary_l


def threshold_abs(img, lo, hi):
    # retain the data within range [lo, hi]
    return np.uint8((img >= lo) & (img <= hi)) * 255


def threshold_rel(img, lo, hi):
    # retain the data within the range [lo, hi] * (max_data - min_data)
    v_min = np.min(img)
    v_max = np.max(img)

    vlo = v_min + (v_max - v_min) * lo
    vhi = v_min + (v_max - v_min) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255


class Thresholding:
    """ This class is for extracting relevant pixels in an image.
    """

    def __init__(self, img_size=(640, 480)):
        self.resize = img_size[0] / 640

    def forward(self, img, lane=2, image=False, debug=False):
        """
        提取特定阈值的像素点

        Parameters:
            img: 原图
            lane: 车道线选择    0 left   1 right   2 dual
            image: 是否显示处理过程
            debug: 是否单步显示
        Returns:
            binary: 阈值化后的二值图
        """
        ##########################################################################################
        """
        Method 1 HLS L Channel
        """
        # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # 0h:hue  1l:luminance   2s:saturation
        # # # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 0h:hue  1s:saturation  2v:value
        # if image:
        #     cv2.imshow('hls', hls)
        #     if debug:
        #         cv2.waitKey(0)
        # left_lane = cv2.inRange(hls, np.array([17, 118, 33]), np.array([75, 191, 154]))
        # if image:
        #     cv2.imshow('dual_lane', left_lane)
        #     if debug:
        #         cv2.waitKey(0)
        # right_lane = threshold_rel(hls[:, :, 1], 0.8, 1.0)
        # dual_lane = left_lane | right_lane
        # dual_lane = threshold_rel(l_channel, 0.6, 1.0)
        
        ##########################################################################################
        
        # hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) 
        # right_lane = threshold_rel(hls_img[:, :, 1], 0.8, 1.0)
        # # right_lane = cv2.bitwise_not(right_lane)

        # black_pixels = (right_lane == 255).sum(axis=0)
        # threshold = 40

        # # 判断是否有反光
        # if np.any(black_pixels > threshold):
        #     print("反光发现")
        #     for i in range(len(black_pixels)):
        #         if black_pixels[i] > threshold:
        #             right_lane[:, i] = 0  # 将mask中对应列的黑色像素变为白色
        # else:
        #     print("无反光")   

        # dual_lane = right_lane 


        
        ##########################################################################################
        """
        Method 2  threshold
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # threshold
        _, dual_lane = cv2.threshold(img, binary_l, 255, binary_type)

        dual_lane = cv2.bitwise_not(dual_lane)

        black_pixels = (dual_lane == 255).sum(axis=0)
        threshold = 30

        # # 判断是否有反光
        # if np.any(black_pixels > threshold):
        #     print("反光发现")
        #     for i in range(len(black_pixels)):
        #         if black_pixels[i] > threshold:
        #             dual_lane[:, i] = 0  # 将mask中对应列的黑色像素变为白色
        # else:
            
            # print("无反光")   

        # ##########################################################################################

         ##########################################################################################
        # """
        # Method 3  threshold
        # """
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # 高斯模糊
        
        # blurred = cv2.GaussianBlur(gray, (9, 9), 0) 
        # # threshold
        # adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        
        # # 形态学处理
        # kernel_size=(7, 7)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        # dual_lane = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

        ##########################################################################################

        ##########################################################################################
        # """
        # Method 4  threshold
        # """
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # # threshold
        # _, dual_lane = cv2.threshold(img, binary_l, 255, binary_type)

        # # 形态学处理
        # kernel_size=(7, 7)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        # dual_lane = cv2.morphologyEx(dual_lane, cv2.MORPH_OPEN, kernel)
        
        ##########################################################################################
        
        
        
        if lane == 0:  # 选择左线
            dual_lane[:, int(275 * self.resize):] = 0  # 将右半图像涂黑
        elif lane == 1:  # 选择右线
            dual_lane[:, :int(375 * self.resize)] = 0  # 将左半图像涂黑

        return dual_lane
