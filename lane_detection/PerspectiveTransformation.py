# -*- coding: utf-8 -*-
import cv2
import numpy as np

try:
    from params import control_points, transform_dst_size, dst_pts
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/jetson/2024')
    from params import control_points, transform_dst_size, dst_pts


class PerspectiveTransformation:
    """
    视角转换类，进行原始视角和俯视视角之间的仿射变换
    """

    def __init__(self, img_size=None):
        """
        初始化

        Parameters:
            img_size: 图片尺寸
        """
        if img_size is None:
            self.img_size = (320, 240)
        else:
            self.img_size = img_size
        self.src = control_points
        self.dst_size = transform_dst_size
        self.dst_pts = dst_pts
        self.M = cv2.getPerspectiveTransform(self.src, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        """
        原始视角  -->  俯视视角

        Parameters:
            img: 原始视角图
            flags: 使用双线性插值

        Returns:
            Image: 俯视视角图
        """
        return cv2.warpPerspective(img, self.M, self.dst_size, flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        """
        俯视视角  -->  原始视角

        Parameters:
            img: 俯视视角图
            flags: 使用双线性插值

        Returns:
            Image: 原始视角图
        """
        return cv2.warpPerspective(img, self.M_inv, self.img_size, flags=flags)


