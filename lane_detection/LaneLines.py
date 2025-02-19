# -*- coding: utf-8 -*-
import cv2

from utils import utils

try:
    from params import *
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/jetson/rc2024')
    from params import *


class LaneLines:
    """
    提取车道线
    """
    def __init__(self, dst_size=(160, 120)):
        """
        初始化

        Parameters:
            img_size_: 图片尺寸
        """
        # np.set_printoptions(precision=6, suppress=True)
        # 超参数
        self.img_size = dst_size
        self.midpoint = self.img_size[0] // 2  # 图像垂直中心
        self.n_windows = n_windows  # 滑动窗口数量
        self.window_height = np.int(self.img_size[1] // n_windows)  # 单个滑动窗口高度
        self.half_height = self.window_height // 2
        self.margin = margin  # 滑动窗口左右宽度
        self.min_pix = min_pix  # 认为滑动窗口中有车道线的最小像素值

        self.canvas = np.zeros((self.img_size[1], self.img_size[0], 3), np.uint8)

        self.left_fit = None
        self.right_fit = None

        self.binary = None
        self.nonzero = None
        self.nonzero_x = None
        self.nonzero_y = None
        self.clear_visibility = True
        self.dir = []

    def forward(self, img, lane=2, image=False, debug=False):
        """
        检测图像中的车道线

        Parameters:
            img: 俯视图阈值化后的二值图
            lane: 车道线选择    0 left   1 right   2 dual
            image: 是否显示处理过程
            debug: 是否单步显示
        Returns:
            out_img: 包含车道线信息的RGB图
        """

        self.extract_features(img)
        img = self.fit_poly(img, lane=lane, image=image, debug=debug)
        gradiant, differ_pix, direction = self.calculate(lane=lane)
        return img, gradiant, differ_pix, direction

    def extract_features(self, img):
        """
        提取二值图中的非0像素点的坐标

        Parameters:
            img: 二值图
        """
        self.nonzero = img.nonzero()
        # self.nonzero = np.where(img == 0)
        self.nonzero_x = np.array(self.nonzero[1])
        self.nonzero_y = np.array(self.nonzero[0])

    def pixels_in_window(self, center):
        """
        返回特定窗口中的像素点

        Parameters:
            center: 窗口中线x坐标

        Returns:
            pixel_x: 窗口中像素点的x坐标
            pixel_y: 窗口中像素点的y坐标
        """
        t_l = (center[0] - self.margin, center[1])
        b_r = (center[0] + self.margin, center[1] + self.window_height)

        coord_x = (t_l[0] <= self.nonzero_x) & (self.nonzero_x <= b_r[0])
        coord_y = (t_l[1] <= self.nonzero_y) & (self.nonzero_y <= b_r[1])
        return self.nonzero_x[coord_x & coord_y], self.nonzero_y[coord_x & coord_y]

    def find_lane_pixels(self, img, lane=2, image=False, debug=False):
        """
        找到属于车道线的像素点

        Parameters:
            img: 俯视图阈值化后的二值图
            lane: 车道线选择    0 left   1 right   2 dual
            image: 是否显示处理过程
            debug: 是否单步显示
        Returns:
            left_x: 左线像素点的x坐标
            left_y: 左线像素点的y坐标
            right_x: 右线像素点的x坐标
            right_y: 右线像素点的y坐标
            out_img: 后续为车道线涂色用的RGB图
        """
        assert (len(img.shape) == 2)

        # 创建一张用来后续做可视化的三通道图片
        out_img = np.dstack((img, img, img))

        # 创建输入的二值图底部一个滑窗高度的区域的直方图
        histogram = np.sum(img[img.shape[0] // 4:, :], axis=0)  # img[h, w, channel]

        # 寻找左线
        if lane == 0:
            # 找到直方图中左起第一个峰，认为是左线的起点
            left_x_base = np.argmax(histogram[:self.midpoint])
            # 当前左线x位置
            left_x_current = left_x_base

            y_current = self.img_size[1]

            # 创建空列表来接收左线像素点的坐标
            left_x, left_y = [], []

            # 遍历每个滑动窗口
            for i in range(self.n_windows):
                y_current -= self.window_height
                center_left = (left_x_current, y_current)
                # if image:
                cv2.rectangle(out_img,
                              (left_x_current - self.margin, y_current),
                              (left_x_current + self.margin, y_current + self.window_height),
                              (200, 200, 0), 2)
                # 找当前窗口中的非0像素点
                good_left_x, good_left_y = self.pixels_in_window(center_left)

                # 将当前窗口中的车道线像素点的坐标记录
                left_x.extend(good_left_x)
                left_y.extend(good_left_y)

                # print('left min_pix', len(good_left_x))
                if len(good_left_x) > self.min_pix:
                    left_x_current = np.int32(np.mean(good_left_x))  # 更新有效窗口的中心位置

            # 可视化
            # if image:
            out_img[left_y, left_x] = [255, 0, 0]
            cv2.line(out_img, (xl_b_pix, self.img_size[1] - 10), (xl_b_pix, self.img_size[1]), (255, 255, 255), 5)
            cv2.line(out_img, (xr_b_pix, self.img_size[1] - 10), (xr_b_pix, self.img_size[1]), (255, 255, 255), 5)
            cv2.imshow('slide windows', out_img)
            cv2.moveWindow("slide windows", 25, 350)
            if debug:
                cv2.waitKey(0)
            return left_x, left_y

        # 寻找右线
        elif lane == 1:
            # 找到直方图右半区域中的第一个峰
            right_x_base = np.argmax(histogram[self.midpoint+40:]) + self.midpoint+40
            right_x_current = right_x_base
            y_current = self.img_size[1]

            right_x, right_y = [], []

            for i in range(self.n_windows):
                y_current -= self.window_height
                center_right = (right_x_current, y_current)
                # if image:
                cv2.rectangle(out_img,
                              (right_x_current - self.margin, y_current),
                              (right_x_current + self.margin, y_current + self.window_height),
                              (255, 200, 0), 2)

                good_right_x, good_right_y = self.pixels_in_window(center_right)

                right_x.extend(good_right_x)
                right_y.extend(good_right_y)

                # print('right min_pix', len(good_right_x))
                if len(good_right_x) > self.min_pix:
                    right_x_current = np.int32(np.mean(good_right_x))
           
            # if image:
            out_img[right_y, right_x] = [0, 0, 255]
            cv2.line(out_img, (xl_b_pix, self.img_size[1] - 10), (xl_b_pix, self.img_size[1]), (255, 255, 255), 5)
            cv2.line(out_img, (xr_b_pix, self.img_size[1] - 10), (xr_b_pix, self.img_size[1]), (255, 255, 255), 5)
            cv2.imshow('slide windows', out_img)
            cv2.moveWindow("slide windows", 25, 350)
            if debug:
                cv2.waitKey(0)
            return right_x, right_y

        # 两条线都寻找
        else:
            left_x_base = np.argmax(histogram[:self.midpoint])
            right_x_base = np.argmax(histogram[self.midpoint:]) + self.midpoint
            left_x_current = left_x_base
            right_x_current = right_x_base
            y_current = self.img_size[1]

            left_x, left_y, right_x, right_y = [], [], [], []

            for i in range(self.n_windows):
                y_current -= self.window_height
                center_left = (left_x_current, y_current)
                center_right = (right_x_current, y_current)
                # if image:
                cv2.rectangle(out_img,
                              (left_x_current - self.margin, y_current),
                              (left_x_current + self.margin, y_current + self.window_height),
                              (150, 200, 0), 2)
                cv2.rectangle(out_img,
                              (right_x_current - self.margin, y_current),
                              (right_x_current + self.margin, y_current + self.window_height),
                              (150, 200, 0), 2)

                good_left_x, good_left_y = self.pixels_in_window(center_left)
                good_right_x, good_right_y = self.pixels_in_window(center_right)

                left_x.extend(good_left_x)
                left_y.extend(good_left_y)
                right_x.extend(good_right_x)
                right_y.extend(good_right_y)

                if len(good_left_x) > self.min_pix:
                    left_x_current = np.int32(np.mean(good_left_x))
                if len(good_right_x) > self.min_pix:
                    right_x_current = np.int32(np.mean(good_right_x))

            # if image:
            out_img[left_y, left_x] = [255, 0, 0]
            out_img[right_y, right_x] = [0, 0, 255]
            cv2.line(out_img, (xl_b_pix, self.img_size[1] - 10), (xl_b_pix, self.img_size[1]), (255, 255, 255), 5)
            cv2.line(out_img, (xr_b_pix, self.img_size[1] - 10), (xr_b_pix, self.img_size[1]), (255, 255, 255), 5)
            cv2.imshow('slide windows', out_img)
            cv2.moveWindow("slide windows", 25, 350)
            if debug:
                cv2.waitKey(0)

            return left_x, left_y, right_x, right_y



    def fit_poly(self, img, lane=2, image=False, debug=False):
        """
        找到二值图中的车道线并画出

        Parameters:
            img: 俯视图阈值化后的二值图
            lane: 车道线选择    0 left   1 right   2 dual
            image: 是否显示处理过程
            debug: 是否单步显示
        Returns:
            out_img: 画了车道线的RGB图
        """
        out_img = self.canvas.copy()

        if lane == 0:
            # 获取左线的像素点坐标
            left_x, left_y = self.find_lane_pixels(img, lane=lane, image=image, debug=debug)

            # print('left_lane_thresh', len(left_y))
            if len(left_y) > left_lane_thresh:
                # 有效左线，进行一阶拟合
                self.left_fit = np.polyfit(left_y, left_x, 1)

            # 生成左线的坐标来画图
            if len(left_y):
                plot_y = np.linspace(np.min(left_y), np.max(left_y), self.n_windows)

                # left_fit_x = self.left_fit[0] * plot_y ** 2 + self.left_fit[1] * plot_y + self.left_fit[2]  # 2nd
                left_fit_x = self.left_fit[0] * plot_y + self.left_fit[1]

                # 可视化
                for i in range(self.n_windows):
                    cv2.circle(out_img,
                               (int(left_fit_x[i]), int(plot_y[i] - self.half_height)), 10,
                               (0, 255, 0), -1)

            return out_img

        elif lane == 1:
            right_x, right_y = self.find_lane_pixels(img, lane=lane, image=image)

            # print('right_lane_thresh', len(right_y))
            if len(right_y) > right_lane_thresh:
                self.right_fit = np.polyfit(right_y, right_x, 1)

            if len(right_y):
                plot_y = np.linspace(np.min(right_y), np.max(right_y), self.n_windows)

                # right_fit_x = self.right_fit[0] * plot_y ** 2 + self.right_fit[1] * plot_y + self.right_fit[2]
                right_fit_x = self.right_fit[0] * plot_y + self.right_fit[1]

                for i in range(self.n_windows):
                    cv2.circle(out_img,
                               (int(right_fit_x[i]), int(plot_y[i] - self.half_height)), 10,
                               (0, 0, 255), -1)
          
            return out_img

        else:
            left_x, left_y, right_x, right_y = self.find_lane_pixels(img, lane=lane, image=image)

            if len(left_y) > left_lane_thresh:
                self.left_fit = np.polyfit(left_y, left_x, 1)
            if len(right_y) > right_lane_thresh:
                self.right_fit = np.polyfit(right_y, right_x, 1)

            if len(left_y) and len(right_y):
                plot_y = np.linspace(np.min(right_y), np.max(right_y), self.n_windows)

                # left_fit_x = self.left_fit[0] * plot_y ** 2 + self.left_fit[1] * plot_y + self.left_fit[2]
                # right_fit_x = self.right_fit[0] * plot_y ** 2 + self.right_fit[1] * plot_y + self.right_fit[2]
                left_fit_x = self.left_fit[0] * plot_y + self.left_fit[1]
                right_fit_x = self.right_fit[0] * plot_y + self.right_fit[1]

                for i in range(self.n_windows):
                    tl = (int(left_fit_x[i]), int(plot_y[i]) - 5)
                    br = (int(right_fit_x[i]), int(plot_y[i]) + self.window_height + 5)
                    cv2.rectangle(out_img, tl, br, (255, 0, 255), -1)

            return out_img

    def calculate(self, lane=2):
        # 计算曲率半径及中线像素偏移
        differ_pix = self.measure(lane=lane)

        # 找到最大斜率
        # print(self.left_fit)
        # print(self.right_fit)
        if lane == 0:
            gradiant = self.left_fit[0]
        elif lane == 1:
            gradiant = self.right_fit[0]
        else:
            gradiant = self.left_fit[0] if abs(self.left_fit[0]) > abs(self.right_fit[0]) else self.right_fit[0]
            
        gradiant = round(gradiant, 5)
        if not gradiant:  # 防止 div 0
            gradiant = 0.00001
        
        # print(gradiant)

        # 确定车道线弯曲方向
        if abs(gradiant) <= straight_threshold:
            self.dir.append('F')
        elif gradiant > 0:
            self.dir.append('L')
        else:
            self.dir.append('R')

        # 平缓化
        if len(self.dir) > 5:
            self.dir.pop(0)

        # 找到出现最多次的方向信息
        direction = max(set(self.dir), key=self.dir.count)

        return gradiant, differ_pix, direction

    def measure(self, lane=2):
        """
        计算俯视视角下的曲率及中心偏差

        Args:
            lane: 车道线选择    0 left   1 right   2 dual

        Returns:
            中心偏差(pix)
        """
        # # 曲率半径计算公式  需用二阶拟合  弃用 应用意义不大 一阶拟合的斜率在本项目中足够
        # x = (left_fit[0])y^2 + (left_fit[1])y + left_fit[2]
        # x' = 2(left_fit[0])y + left_fit[1]
        # x'' = 2left_fit[0]
        # k = |x''| / ((1+(x')^2)^(1.5))
        # R = 1 / k

        # 计算视野中心距道路中心的偏移(选取底部为参考)
        if lane == 0:  # 左线
            # 计算曲率半径
            # curveR = ((1 + (2 * left_fit[0] * y_real + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
            # xl = np.dot(self.left_fit, [dst[1] ** 2, dst[1], 1])  # 2nd
            xl = np.dot(self.left_fit, [transform_dst_size[1], 1])
            differ = -((xr_b_pix + xl_b_pix)/2 -xl - 119 -2.5) -6.3 -3.5
            print(f"differ = {differ}")
        elif lane == 1:  # 右线

            # curveR = ((1 + (2 * right_fit[0] * y_real + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
            # xr = np.dot(self.right_fit, [dst[1] ** 2, dst[1], 1])

            xr = np.dot(self.right_fit, [transform_dst_size[1], 1])
            differ = xr - (xr_b_pix + xl_b_pix)/2 - 108 +1.6 -1
            print(differ)
        else:  # 双线
            # curveR_l = ((1 + (2 * left_fit[0] * y_real + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
            # curveR_r = ((1 + (2 * right_fit[0] * y_real + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
            # curveR = min(curveR_l, curveR_r)
            # xl = np.dot(self.left_fit, [dst[1] ** 2, dst[1], 1])
            # xr = np.dot(self.right_fit, [dst[1] ** 2, dst[1], 1])

            xl = np.dot(self.left_fit, [transform_dst_size[1], 1])
            xr = np.dot(self.right_fit, [transform_dst_size[1], 1])
            differ = (transform_dst_size[0] // 2 - (xl + xr) // 2)

        return differ
