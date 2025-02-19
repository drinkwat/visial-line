# -*- coding: utf-8 -*-
import numpy as np
try:
    from utils.utils import *
except ModuleNotFoundError:
    from utils import *


# #####################################################################
# camera
# ########
camera_fixed_config = CameraConfig(name='Fixed Cam',   # 不修改 须与变量名对应
                                  camera_id=0,  # 相机id
                                  brightness=0,  # 亮度 -64~(0)~64
                                  contrast=4,  # 对比度 0~(4)~95
                                  exposure=0, # 0 自动曝光  1~10000 手动曝光值
                                  img_size=(320, 240)  # 画幅
                                 )
camera_mobile_config = CameraConfig(name='Mobile Cam',  # 不修改 须与变量名对应
                                   camera_id=1,  # 相机id
                                   brightness=0,  # 亮度-64~(0)~64
                                   contrast=4,  # 对比度 0~(4)~95
                                   exposure=0,  # 0 自动曝光  1~10000 手动曝光值
                                   img_size=(320, 240)  # 画幅
                                  )

"""
相机内参
可以使用ros中的标定工具camera_calibration来获取
"""
# camera_matrix
# 从  ~/.ros/camera_info/head_camera.yaml >> camera_matrix >> data
# [fx, 0, cx, 0, fy, cy, 0, 0, 1]
# 转为  np.array()
# [[fx, 0, cx],
#  [0, fy, cy],
#  [0, 0, 1]]
camera_matrix = np.array([[411.3261791696689, 0.0, 323.6570089289733],  # [fx, 0, cx]s
                          [0.0, 408.1251597540983, 238.7885974162409],  # [0, fy, cy]
                          [0.0, 0.0, 1.0]],
                         dtype=np.float64)
# distortion_coefficients
# 从  ~/.ros/camera_info/head_camera.yaml >> distortion_coefficients >> data
# [k1, k2, p1, p2, 0]
# 转为  np.array()
# [k1, k2, p1, p2]
distortion_coefficients = np.array([-0.3085708052694404, 0.07981579355064194, 0.0008839337992127878, -0.0009765228024578548],
                                   dtype=np.float64)
# projection_matrix
# 从  ~/.ros/camera_info/head_camera.yaml >> projection_matrix >> data
# [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
# 转为  np.array()
# [[fx, 0, cx],
#  [0, fy, cy],
#  [0, 0, 1]]
projection_matrix = np.array([[316.1893310546875, 0.0, 323.9889318368223],
                              [0.0,        358.1781311035156, 239.0412111827609],
                              [0.0,               0.0,        1.0]],
                             dtype=np.float64)
                             
# #####################################################################


# #####################################################################
# chassis
# ########
# 模式位
disarm    = 0  # 上锁   线速度为0
arm       = 1  # 解锁   线速度为1
vel_mode  = 2  # 速度模式   线速度为巡线速度


# 以下参数配合/chassis/chassis_control.py  系数校准  获取
# 通过里程计获得线位移
# 通过指南针获得角位移
l_0x40 = 0.63  # 线位移 m
a_0x40 = 27.0  # 角位移 °
# vel_2_0x40 = 64 / l_0x40  # 分母为 0x40/(+64)校准时 1s内行驶的距离 m
# omega_2_0x40 = 64 / a_0x40  # 分母为 最大值0x40/(+64)校准时 1s内旋转的角度 °
vel_2_0x40 = 1.0 / l_0x40  # 分母为 0x40/(+64)校准时 1s内行驶的距离 m
omega_2_0x40 = 0.465 / a_0x40  # 分母为 最大值0x40/(+64)校准时 1s内旋转的角度 °

# #####################################################################


# #####################################################################
# lane_detection
# ######
lane_list = ["left", "right", " dual"]

"""
control points
"""
# 实际控制点离开车道线的像素距离 pix
control_pts_offset = 35

# 存储坐标点
control_points = np.float32(np.load('/home/iflytek/visial_line/perspective.npy'))

transform_dst_size = (320, 240)
dst_pts = np.float32([(0,                     0),
                      (transform_dst_size[0], 0),
                      (transform_dst_size[0], transform_dst_size[1]),
                      (0,                     transform_dst_size[1])])

"""
thresholding
"""
# 笃南
binary_type = 1  # 0 cv2.THRESH_BINARY(黑底白线->黑底白线 / 白底黑线->白底黑线)   1 cv2.THRESH_BINARY_INV(白底黑线->黑底白线 / 黑底白线->白底黑线)
# 以下 二值化 参数配合/utils/thresh_binary.py获取
# binary_l = 177
# binary_l = 176  # 光线较暗
binary_l = 177  # 光线较亮
# binary_type = 0  # 0 cv2.THRESH_BINARY(黑底白线->黑底白线 / 白底黑线->白底黑线)   1 cv2.THRESH_BINARY_INV(白底黑线->黑底白线 / 黑底白线->白底黑线)
# # 以下 二值化 参数配合/utils/thresh_binary.py获取
# binary_l = 137

"""
LaneLines
"""
# 滑动窗口数量
n_windows = 9
# 窗口左右覆盖范围 pix
margin = 40
# 单个滑动窗口内确认出现车道线所需的最少像素点数 pix
min_pix = 50

# 二值图中左/右线像素点数阈值 pix
left_lane_thresh = 100
right_lane_thresh = 100

"""
curvature
"""
# 以下 车道线 参数配合/utils/lane_pix.py获取
# 对正车道线时 俯视视角中 左/右车道线底部的x轴像素坐标
# 笃南
xl_b_pix = 56
xr_b_pix = 279
# 笃南
x_real = 0.40  # m  左右  (固定值 与地图尺寸相关)
y_real = 0.41  # m  前后  (测量值 与控制点选取相关)
# 像素距离到实际距离的转换系数
x_pix2m = x_real / dst_pts[1][0] - dst_pts[0][0]
y_pix2m = y_real / dst_pts[2][1] - dst_pts[1][1]

# 认为车道线为直线时 车道线一次拟合的斜率的绝对值 的阈值
straight_threshold = 0.04
# straight_threshold = 0.10


"""
Kp
"""
max_vx = 0.2    # 表征最高限度 0.2 ，控制线速度
kp_x = 0.1    # 表征弯道降速 0.001，
kp_y = 100        # 表征横向纠偏 3
kp_w = -800      # 表征弯道转速 120

buffer_size = 1  # 视选取的控制点超前车身的情况 适当增加缓存的指令数量 以达到滞后控制的效果

# #####################################################################


