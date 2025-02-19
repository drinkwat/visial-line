# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sharedmem
import time
try:
    from params import camera_fixed_config, camera_mobile_config, my_print, disarm, vel_mode
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/ucar/visial_line')
    from params import camera_fixed_config, camera_mobile_config, my_print, disarm, vel_mode


def camera(flag_task_v, img_mem_v,
           flag_lane_v, cmd_chassis_v,
           camera_config_v, record_v=False):
    """
    采集指定编号相机的帧

    Args:
        flag_task_v:        任务进行标志位 (共享)
        img_mem_v:          相机捕获的帧 (共享)
        flag_lane_v:        车道线行驶外部中断 (共享)
        cmd_chassis_v:      底盘控制 (共享)
        camera_config_v:    相机配置 (包含相机编号 画幅 亮度 对比度 曝光度)  相机编号* (ls -l /dev/video*  约定 0定 1动)
        record_v:           是否录像 (默认不录)
    """
    my_print("这里是读取机动位摄像头进程" if camera_config_v.camera_id else "这里是读取固定位摄像头进程")

    cap = cv2.VideoCapture(camera_config_v.camera_id)
    resize = False
    if camera_config_v.img_size[0] != 320 or camera_config_v.img_size[1] != 240:
        resize = True
    if isinstance(camera_config_v.camera_id, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config_v.img_size[0])  # 设置宽度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config_v.img_size[1])  # 设置高度
        cap.set(cv2.CAP_PROP_BRIGHTNESS, camera_config_v.brightness)  # 设置亮度
        cap.set(cv2.CAP_PROP_CONTRAST, camera_config_v.contrast)  # 设置对比度
        if camera_config_v.exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0~2.6手动曝光  2.6~4自动曝光
            cap.set(cv2.CAP_PROP_EXPOSURE, camera_config_v.exposure)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 设置视频流格式
        my_print(f"Setting {camera_config_v.name} to "
                 f"Width {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}  "
                 f"Height {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}  "
                 f"Brightness {cap.get(cv2.CAP_PROP_BRIGHTNESS)}  "
                 f"Contrast {cap.get(cv2.CAP_PROP_CONTRAST)} "
                 f"Exposure {'Manual ' + str(cap.get(cv2.CAP_PROP_EXPOSURE)) if camera_config_v.exposure else 'Auto'} ")

    else:
        my_print("Wrong Camera ID", 'err')
        return
    
    out = None

    if record_v and isinstance(camera_config_v.camera_id, int):
        # # 录制视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频格式
        fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 视频宽高
        out = cv2.VideoWriter(f'video_{int(time.time())}.mp4', fourcc, fps, size)

    cv2.namedWindow(camera_config_v.name)

    while flag_task_v.value and cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        if not ret:
            my_print(f'Fail to Load Stream of Camera {camera_config_v.camera_id}', 'err')
            break

        if record_v:
            out.write(img)  # 录制视频

        img_mem_v[:] = img.copy()

        if resize:
            img = cv2.resize(img, (320, 240))

        cv2.imshow(camera_config_v.name, img)
        cv2.moveWindow(camera_config_v.name, 800, 5 if camera_config_v.name == "Mobile Cam" else 350)

        key = cv2.waitKey(1) & 0xff
        if key == ord(' '):  # 暂停 中止检测
            flag_lane_v.value = 0  # 停止车道线检测
            cmd_chassis_v[:] = [vel_mode, 0, 0, 0]  # 停车
            img = np.zeros((camera_config_v.img_size[1], camera_config_v.img_size[0], 3))  # 清空画面，防止误识别
            img_mem_v[:] = img.copy()
            if resize:
                img = cv2.resize(img, (320, 240))
            cv2.imshow(camera_config_v.name, img)
            cv2.waitKey(0)
            flag_lane_v.value = 1  # 恢复车道线检测
        elif key == 27:
            cmd_chassis_v[:] = [vel_mode, 0, 0, 0]  # 停车
            break
  
    cap.release()
    cv2.destroyAllWindows()
    if record_v:
        out.release()
    cmd_chassis_v[:] = [disarm, 0, 0, 0]  # 停车
    time.sleep(.5)
    my_print("读取机动位摄像头进程结束" if camera_config_v.camera_id else "读取固定位摄像头进程结束")
    flag_task_v.value = 0


if __name__ == '__main__':
    from multiprocessing import Process, Array, Value

    if len(sys.argv) <= 1:
        exit("Please input camera id\nExample: python3 camera.py 0")

    try:
        cam = int(sys.argv[1])
    except ValueError:
        exit("Camera id should be int\nExample: python3 camera.py 0")

    if cam not in [0, 1, 2]:
        exit("Camera id should be among [0, 1, 2] in this porject\nExample: python3 camera.py 0")
        
    # 创建蒙版
    mask_mobile_left = np.zeros((camera_mobile_config.img_size[1], camera_mobile_config.img_size[0], 3), np.uint8)
    mask_mobile_grab_with_claw = np.zeros((camera_mobile_config.img_size[1], camera_mobile_config.img_size[0], 3), np.uint8)
    mask_mobile_put_no_claw = np.zeros((camera_mobile_config.img_size[1], camera_mobile_config.img_size[0], 3), np.uint8)
    dual_size = [camera_fixed_config.img_size[0]+camera_mobile_config.img_size[0]+50,
                 max(camera_fixed_config.img_size[1], camera_mobile_config.img_size[1])]  # 机动位 + 固定位
    mask_dual_put_no_claw = np.zeros((dual_size[1], dual_size[0], 3), np.uint8)
    
    # 绘制蒙版通过区域  蒙版初始化后为全黑 通过区域内涂白 后续图像和蒙板进行与操作时 涂白的区域内的图像可以保留
    # 机动位 灰度节点处左视 检测树上有无水果
    mask_mobile_left[40:180, 40:180] = np.ones((140, 140, 3), np.uint8) * 255  # [yl:yh, xl:xr]
    # 机动位 果树前正视 检查水果位置 含爪
    mask_mobile_grab_with_claw[20:240, 50:230] = np.ones((220, 180, 3), np.uint8) * 255
    # 机动位 前往放置时 未避免爪中水果干扰 无爪
    mask_mobile_put_no_claw[:160, :] = np.ones((160, camera_mobile_config.img_size[0], 3), np.uint8) * 255
    # 机动位 + 固定位 前往放置时 未避免爪中水果干扰  无爪
    mask_dual_put_no_claw[:160, :camera_fixed_config.img_size[0]] = np.ones((160, camera_fixed_config.img_size[0], 3), np.uint8) * 255
    mask_dual_put_no_claw[:, camera_fixed_config.img_size[0]+50:] = np.ones((dual_size[1], camera_mobile_config.img_size[0], 3), np.uint8) * 255
    
    # 创建图像共享内存
    img_mem_fixed = sharedmem.empty((camera_fixed_config.img_size[1], camera_fixed_config.img_size[0], 3), np.uint8)  # 固定位图像帧 初始化为全黑
    img_mem_mobile = sharedmem.empty((camera_mobile_config.img_size[1], camera_mobile_config.img_size[0], 3), np.uint8)  # 机动位图像帧 初始化为全黑
    img_dual = np.zeros((dual_size[1], dual_size[0], 3), np.uint8)  # 机动位 + 50间隔 + 固定位

    flag_task = Value('i', 1)
    flag_lane = Value('i', 1)  # 车道线辅助行驶标志(默认开启)
    cmd_vel = Array('d', 4)  # 底盘控制 [mode linear.x linear.y linear.z angular.z]

    record = False

    # 固定位视频帧采集子进程
    p_video_fixed = Process(target=camera,
                            args=(flag_task, img_mem_fixed,
                                  flag_lane, cmd_vel,
                                  camera_fixed_config, record))
    # 机动位视频帧采集子进程
    p_video_mobile = Process(target=camera,
                             args=(flag_task,img_mem_mobile,
                                   flag_lane, cmd_vel,
                                   camera_mobile_config, record))
   
    if cam in [0, 2]:
        p_video_fixed.start()
    if cam in [1, 2]:
        p_video_mobile.start()
    
    mask_type = 0
    
    while flag_task.value:
        # 选择检测视频源
        if cam == 0:  # 固定位
            img = np.array(img_mem_fixed, np.uint8)
        elif cam == 1:  # 机动位
            img = np.array(img_mem_mobile, np.uint8)
        else:
            img = img_dual.copy()
            img[:camera_mobile_config.img_size[1], :camera_mobile_config.img_size[0]] = np.array(img_mem_mobile, np.uint8)
            img[:camera_fixed_config.img_size[1], camera_mobile_config.img_size[0]+50:] = np.array(img_mem_fixed, np.uint8)
        
        # 是否需要提取roi
        # 1 机动位 灰度节点处左视 检测树上有无水果  2 机动位 果树前正视 检查水果位置 含爪
        # 3 机动位 前往放置时 未避免爪中水果干扰 无爪  4 机动位 + 固定位 前往放置时 未避免爪中水果干扰  无爪
        if mask_type == 1 and cam == 1:
            img = cv2.bitwise_and(img, mask_mobile_left)
        elif mask_type == 2 and cam == 1:
            img = cv2.bitwise_and(img, mask_mobile_grab_with_claw)
        elif mask_type == 3 and cam == 1:
            img = cv2.bitwise_and(img, mask_mobile_put_no_claw)
        elif mask_type == 4 and cam == 2:
            img = cv2.bitwise_and(img, mask_dual_put_no_claw)
                
        cv2.imshow('img', img)
        cv2.moveWindow('img', 25, 5)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('1'):
            mask_type = 1
        elif key == ord('2'):
            mask_type = 2
        elif key == ord('3'):
            mask_type = 3
        elif key == ord('4'):
            mask_type = 4
            

    flag_task.value = 0

    try:
        p_video_fixed.join()
    except (AssertionError, NameError):
        pass
    try:
        p_video_mobile.join()
    except (AssertionError, NameError):
        pass
