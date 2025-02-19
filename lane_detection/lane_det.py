# -*- coding: utf-8 -*-
import time
import cv2
try:
    from params import *
    from lane_detection.Thresholding import *
    from lane_detection.PerspectiveTransformation import *
    from lane_detection.LaneLines import *
except ModuleNotFoundError:
    import sys

    sys.path.append('/home/iflytek/visial_line')
    from params import *
    from Thresholding import *
    from PerspectiveTransformation import *
    from LaneLines import *

"""
进行车辆道路线的检测和跟踪。
"""
class FindLaneLines:
    """
    寻找车道线
    """
    def __init__(self, img_size_=(320, 240)):
        """ Init Application"""
        self.img_size = img_size_
        self.thresholding = Thresholding(img_size=img_size_)  # 二值化
        self.transform = PerspectiveTransformation(img_size=img_size_)  # 透视变换
        self.lane_lines = LaneLines(dst_size=transform_dst_size) # 未知，猜测传入的参数为去畸变参数

        # 读取了左转，右转，直走的图片
        self.left_curve_img = cv2.imread('/home/iflytek/visial_line/utils/left_turn.png',
                                         cv2.IMREAD_UNCHANGED)
        self.right_curve_img = cv2.imread('/home/iflytek/visial_line/utils/right_turn.png',
                                          cv2.IMREAD_UNCHANGED)
        self.keep_straight_img = cv2.imread('/home/iflytek/visial_line/utils/straight.png',
                                            cv2.IMREAD_UNCHANGED)
        
        # 缩放图像，使其新的 width（宽度）和 height（高度）分别等于原图像尺寸与640和480的比例。
        self.left_curve_img = cv2.resize(self.left_curve_img, (0, 0), fx=self.img_size[0] / 640, fy=self.img_size[1] / 480)
        self.right_curve_img = cv2.resize(self.right_curve_img, (0, 0), fx=self.img_size[0] / 640, fy=self.img_size[1] / 480)
        self.keep_straight_img = cv2.resize(self.keep_straight_img, (0, 0), fx=self.img_size[0] / 640, fy=self.img_size[1] / 480)

        # 获取调整大小后的左转图像的尺寸（宽度、高度和通道数）
        self.sign_size = self.left_curve_img.shape

        # 定义新的宽度 W 和高度 H 的值
        self.W = 125 + self.sign_size[0]
        self.H = 150 + self.sign_size[1]


    def forward(self, img, lane, image=True, debug=False):

        image = True
        if image:
            cv2.imshow('undistort', img)
            if debug:
                cv2.waitKey(0)

        # 图像去畸变
        # 因畸变主要在固定位相机中出现，而由于包括 插usb的先后顺序在内 的各种原因 可能导致相机id顺序并不为本代码约定的顺序
        # 故在此处确定固定位相机视频帧时进行去畸变
        # 使用undistort方法去畸变
        # img = cv2.undistort(src=img, cameraMatrix=camera_matrix, distCoeffs=distortion_coefficients,
        #                     dst=None, newCameraMatrix=projection_matrix)
        # if image:
        #     cv2.imshow('distort', img)
        #     if debug:
        #         cv2.waitKey(0)
        
        # 复制一张原图
        out_img = np.copy(img)

        # 把指定的控制点在图像上标记出来
        # 将 debug 设置为真，图像会停留在屏幕上，等待任意键的输入来关闭图像窗口，
        # 否则，图像窗口会在短暂的时间后自动关闭。
        if image:
            # 将四个预设点添加到图像中
            # 并将图像打印到窗口
            with_dot = np.copy(img)
            for c in range(4):
                cv2.circle(with_dot, control_points[c].astype(int), 5, (255, 0, 255), -1)
                cv2.line(with_dot,
                         control_points[c].astype(int),
                         control_points[c+1 if c+1 < 4 else 0].astype(int),
                         (255, 255, 0), 1)
            cv2.imshow('Ctrl Pts', with_dot)
            if debug:
                cv2.waitKey(0)
 
        # 因为是白底，故先进行阈值过滤再转换视角，防止视角转换时出现黑边影响后续
        img = self.thresholding.forward(img, lane=lane, image=image, debug=debug)
        if image:
            cv2.imshow(f'threshold {lane_list[lane]} lane', img)
            if debug:
                cv2.waitKey(0)

        # 视角转换
        img = self.transform.forward(img)
        if image:
            cv2.imshow('top down view', img)
            if debug:
                cv2.waitKey(0)

        # 车道检测
        img, gradiant, differ_pix, direction = self.lane_lines.forward(img, lane=lane, image=image, debug=debug)
        if image:
            cv2.imshow('lane in top down view', img)
            if debug:
                cv2.waitKey(0)

        # 视角转换回来
        img = self.transform.backward(img)
        if image:
            cv2.imshow('lane in original view', img)
            if debug:
                cv2.waitKey(0)

        # 车道分割 叠加至原图
        out_img = cv2.addWeighted(out_img, 1, img, 0.8, 0)

        # # 添加车道信息
        differ = differ_pix * x_pix2m   # 像素距离转换成实际距离
        out_img = self.draw_osd(out_img, differ, direction, lane=lane)
        return out_img, gradiant, differ
    
    def draw_osd(self, out_img, differ, direction, lane):
        # 画车道线中线,红线
        cv2.line(out_img,
                 (int((control_points[3][0] + control_points[2][0])//2 - differ/x_pix2m), self.img_size[1] - 40),
                 (int((control_points[3][0] + control_points[2][0])//2 - differ/x_pix2m), self.img_size[1]),
                 (0, 0, 255), 5)

        # 画视野中线，蓝线
        cv2.line(out_img,
                 (int(control_points[3][0] + control_points[2][0]) // 2 - 40, self.img_size[1] - 20),
                 (int(control_points[3][0] + control_points[2][0]) // 2 - 40, self.img_size[1]),
                 (255, 0, 0), 5)
        
        # 添加相关信息
        widget = np.copy(out_img[30:self.H, :self.W])
        widget = widget // 1.2
        widget[0, :] = [0, 0, 255]
        widget[-1, :] = [0, 0, 255]
        widget[:, 0] = [0, 0, 255]
        widget[:, -1] = [0, 0, 255]
        out_img[30:self.H, :self.W] = widget

        # 添加可视化箭头
        if direction == 'L':
            msg = "Left Curve Ahead"
            out_img = utils.overlayPNG(out_img, self.left_curve_img,
                                       [self.W // 2 - self.sign_size[0] // 2, 35])

        elif direction == 'R':
            msg = "Right Curve Ahead"
            out_img = utils.overlayPNG(out_img, self.right_curve_img,
                                       [self.W // 2 - self.sign_size[0] // 2, 35])

        else:
            msg = "Keep Straight Ahead"
            out_img = utils.overlayPNG(out_img, self.keep_straight_img,
                                       [self.W // 2 - self.sign_size[0] // 2, 35])

        cv2.putText(out_img, msg, org=(5, 50 + self.sign_size[1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255, 0, 0), thickness=2)

        cv2.putText(out_img, f"{lane_list[lane]} Lane Keeping",
                    org=(5, 70 + self.sign_size[1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0), thickness=2)

        cv2.putText(out_img, "{:.2f} m off center".format(-differ),
                    org=(5, 90 + self.sign_size[1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255, 255, 255), thickness=2)
        
        return out_img
    
    def process_frame(self, img, lane, image=False):
        """
        处理视频帧    配合lane_detection用

        Args:
            img:            视频帧
            lane:           车道线选择   0 左  1 右  2 双

        Returns:
            处理后图像, 曲率半径, 中心偏差
        """
        out_img, gradiant, differ = self.forward(img, lane, image=image, debug=False)

        return out_img, gradiant, differ


def lane_detection(flag_task_l, img_mem_front_l, flag_lane_l, number_lane_l, cmd_chassis_l, image=False):
    """"
    车道检测线主函数
    参数：
    flag_task_l： 任务进行标志位
    img_mem_front_l： 接受从摄像头获取的图像
    flag_lane_l：未知，初始值为0，进程全部打开后为1
    number_lane_l：选择车道线
    cmd_chassis_l：小车速度状态，多进程共享变量。这个变量可以在多个进程之间共享并更新，使得不同的进程能够读取到同一变量的最新值。
    image：用于判断图像是否为空，在FindLaneLines.forward方法中使用
    处理摄像头获取的图像，并将处理后的图像实时显示，
    计算曲率半径, 中心偏差
    并通过二者计算小车的前进速度，横向修正速度,旋转速度
    并将速度存储至cmd_chassis_l
    """
    # 创建巡线实例
    findLaneLines_l = FindLaneLines(img_size_=camera_fixed_config.img_size)
    keep_lane = False
    
    my_print("这里是车道线检测进程")
    cv2.namedWindow("Lane")
    cv2.moveWindow("lane", 425, 350)

    # 作为缓冲队列使用。它存储了未来要送入底盘控制的控制命令
    # [vel_mode, 0, 0, 0]
    # [车辆的速度模式, 前进速度，横向修正速度,旋转速度]
    # buffer_size = 1 缓冲区大小

    cmd_buffer = [[vel_mode, 0, 0, 0]] * buffer_size

    # flag_lane_l.value 用于判断车辆此时是否需要保持在车道上行驶。
    # keep_lane 用于控制车辆在退出车道保持后是否需要停车，将速度改为0，然后延时，实现停车
    while flag_task_l.value:

        if flag_lane_l.value:
            keep_lane = True

            # 将存储在内存中的图像img_mem_front_l提取出来，并将其转化为8位无符号整数的NumPy数组格式。
            img = np.array(img_mem_front_l, np.uint8)

            try:
                # 记录当前的时间，用于后续计算函数的运行时间
                t_l_0 = time.time()

                # 传入图像，车道线选择，以及输入图像不为空
                # 返回 处理后图像, 曲率半径, 中心偏差
                out_img, gradiant, differ = findLaneLines_l.process_frame(img, number_lane_l.value, image=image)
                # print(number_lane_l.value)

                # # 选择车道线
                # if gradiant > 0.21:
                #     number_lane_l.value = 1
                # else:
                #     number_lane_l.value = 2
                # 计算帧率
                fps_lane = round(1 / (time.time() - t_l_0))
                # 将帧率添加在图像上
                cv2.putText(out_img, f'FPS:{fps_lane}', (7, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (125, 0, 255), 2)
                # 显示处理后的图像和帧率
                cv2.moveWindow("Lane", 425, 350)
                cv2.imshow('Lane', out_img)
                
        
                # cmd_future = [vel_mode,                                                   # 模式
                #               min(round(max_vx - 0.2 + kp_x / abs(gradiant), 2), max_vx), # 前进速度  有弯则适当减速 <max_vx
                #               round(kp_y * differ, 2),                                    # 横向对正车道中线
                #               round(kp_w * gradiant, 2)]   
                # print(differ)
                cmd_future = [vel_mode,                                                   # 模式
                              min(round(max_vx - 0.2 + kp_x / abs(gradiant), 2), max_vx), # 前进速度  有弯则适当减速 <max_vx
                              round(kp_y * 0, 2),                                    # 横向对正车道中线
                              round(kp_w * differ, 2)]                                  # 转弯速度
                # 更新指令的缓存cmd_buffer，并将新的命令更新到cmd_chassis_l
                if flag_lane_l.value:
                    cmd_chassis_l[:] = cmd_buffer[0]
                    # print("cmd vel lane", cmd_chassis_l[:])
                    cmd_buffer.pop(0)
                    cmd_buffer.append(cmd_future)
                else:
                    cmd_chassis_l[:] = [vel_mode, 0, 0, 0]
            except (TypeError, ZeroDivisionError):
                pass
            key = cv2.waitKey(1) & 0xff

            # 按下Esc，强制停车，退出巡线模式
            if key == 27:
                cmd_chassis_l[:] = [vel_mode, 0, 0, 0]
                time.sleep(1)
                flag_lane_l.value = 0
                flag_task_l.value = 0
                break
            # 按下空格，强制停车，等待下一个CV事件
            elif key == ord(' '):
                cmd_chassis_l[:] = [vel_mode, 0, 0, 0]
                cv2.waitKey(0)

            # 手动切换车道线选择
            elif key == ord('a'):
                number_lane_l.value = 0
            elif key == ord('d'):
                number_lane_l.value = 1
            elif key in (ord('w'), ord('s')):
                number_lane_l.value = 2

        else:
            if keep_lane:  # 确保退出车道线行驶后能停车
                cmd_chassis_l[:] = [vel_mode, 0, 0, 0]
                keep_lane = False

            time.sleep(0.1)


if __name__ == "__main__":
    pass
    """
    有底盘测试
    初始化进程变量
    开启巡线进程
    同时设置标志位结束巡线进程
    """
    # 创建和管理在不同进程之间共享的NumPy数组
    import sharedmem

    # Process：多进程
    # Value，Array:共享内存
    from multiprocessing import Process, Value, Array
    import sys
    
    sys.path.append('../')
    # 导入自定义的camera 模块
    from camera.camera import camera
    # 导入自定义的底盘控制模块
    from chassis.chassis_control import chassis_control

    # 检查用户是否输入参数
    if len(sys.argv) <= 1:
        exit("Please input bool value to indicate whether with chassis or not\nExample: python3 lane_detection.py 1")

    # 将输入的参数转化为整数，赋值给 with_chassis 变量
    with_chassis = int(sys.argv[1])

    # 检查用户输入的参数是否在 [0, 1] 中，如果不在 [0, 1] 中，那么终止程序，并给出错误提示。
    if with_chassis not in [0, 1]:
        exit("Bool value should be among [0, 1] in this porject\nExample: python3 lane_detection.py 1")
    
    # 创建进程间通信变量并初始化

    #  创建一个空的在进程间共享的多维数组，用于存储从摄像头获取的图像。
    img_mem_fixed = sharedmem.empty((camera_fixed_config.img_size[1], camera_fixed_config.img_size[0], 3), np.uint8)  # 固定位图像帧 初始化为全黑
    
    # 始终为1，若 flag_task 改为0，杀死巡线所有进程，巡线结束
    flag_task = Value('i', 1)  # 任务进行标志位(默认开启)
    
    # 初始值为0，进程全部打开后为1
    flag_lane = Value('i', 0)  # 车道线辅助行驶标志(默认开启)
    number_lane = Value('i', 1)  # 车道线选择   0 左  1 右  2 双
    
    # 四元数，控制底盘速度
    cmd_chassis = Array('d', 4)  # 底盘控制 [1 linear.x linear.y linear.z angular.z] v = ωr
    cmd_chassis[0] = arm  # 解锁
    # 减速
    slow_down = Value('i', 0)
    
    # 固定位视频帧采集子进程
    # 创建一个进程，其目标函数是camera，参数是一些共享变量和配置文件。
    p_video_fixed = Process(target=camera,
                            args=(flag_task, img_mem_fixed,
                                  flag_lane, cmd_chassis,
                                  camera_fixed_config, False))
    
    # 车道线子进程
    p_lane = Process(target=lane_detection,
                     args=(flag_task, img_mem_fixed, flag_lane, number_lane, cmd_chassis, True))
    
    # 底盘控制子进程
    p_chassis = Process(target=chassis_control,
                        args=(flag_task, cmd_chassis, slow_down))
    
    my_print("开启固定位摄像头读取帧进程")
    p_video_fixed.start()
    
    # 这里类似于我们的base_driver.launch
    if with_chassis:
        my_print("开启底盘行驶进程")
        p_chassis.start()
        time.sleep(1)
    
    my_print("开启车道线检测进程")
    p_lane.start()
    
    flag_lane.value = 1
    while flag_task.value:
        time.sleep(1)
    
    try:
        p_video_fixed.join()
    except (AssertionError, NameError):
        pass
    try:
        p_lane.join()
    except (AssertionError, NameError):
        pass
    try:
        p_chassis.join()
    except (AssertionError, NameError):
        pass
