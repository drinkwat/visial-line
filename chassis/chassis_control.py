#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from multiprocessing import Process, Value, Array
import time

try:
    from params import my_print, disarm, arm, vel_mode, vel_2_0x40, omega_2_0x40
except ModuleNotFoundError:
    import sys
    sys.path.append('/home/ucar/visial_line')
    from params import my_print, disarm, arm, vel_mode, vel_2_0x40, omega_2_0x40


def chassis_control(flag_task_c, cmd_chassis_c, slow_down_c, on_position=None):
    """
    底盘控制

    Args:
        flag_task_c:        任务开启标志位 (共享)
        cmd_chassis_c:      底盘控制 (共享)
        slow_down_c:        检测到eor后减速以便于停车 (共享)
        on_position:        灰度控制到位标志 (共享) (默认未到位)

    Returns:

    """
    my_print("这里是行驶进程")

    # 初始化ROS node
    rospy.init_node('linetrack', anonymous=True)
    cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size = 10)

    # 实现引擎启动操作
    msg = Twist()
    last_instruction = []

    while flag_task_c.value:
      
        if cmd_chassis_c[:] == last_instruction:
            continue
            
        if cmd_chassis_c[0] in [disarm, arm]:
            msg.linear.x = cmd_chassis_c[0]  
            
            my_print("Engine Shutdown" if cmd_chassis_c[0] == disarm else "Engine Start", "warn")
            cmd_chassis_c[:] = [8, 0, 0, 0]
            
        # velocity control
        elif cmd_chassis_c[0] == vel_mode:
            
            vel = [cmd_chassis_c[1] * vel_2_0x40, 
                   cmd_chassis_c[2] * vel_2_0x40, 
                   cmd_chassis_c[3] * omega_2_0x40]
            if slow_down_c.value:
                vel[0] = vel[0] * 0.7
            if vel[0] < 0:
                vel[2] = -vel[2]
            
            msg.linear.x = vel[0]
            msg.angular.z = vel[2]
            cmd_vel_pub.publish(msg)
            time.sleep(.001)

        last_instruction = cmd_chassis_c[:]

    my_print("行驶进程结束")


def drive_calib(mode):
    """
    速度校准 (认为线速度/角速度与电压成近似线性关系 取半电压为基准运行1s 测量实际运行情况)

    Args:
        mode:               校准模式  linear 线速度校准   angular 角速度校准
    """

    # 初始化 ROS 节点
    rospy.init_node('drive_calib', anonymous=True)

    # 创建发布者，话题的名称是'/cmd_vel'，消息的类型是Twist
    cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist,queue_size = 10)

    # 创建 Twist 类型消息对象
    twist = Twist()

    # Get the current time
    start_time = time.time()

    while True:
        if mode == 'linear':
            twist.linear.x = 0.64  # 设置线速度
        elif mode == 'angular':
            twist.angular.z = -0.64  # 设置角速度 以方便配合手机指南针功能中的角度定义使用右转

        # 发布消息
        cmd_vel_pub.publish(twist)

        # 检查是否已经运行了1s
        if time.time() - start_time > 1:
            break

        # 等待一小段时间，然后开始下一次循环
        time.sleep(0.01)

    # 停止运动，发布停止的消息
    twist.linear.x = 0
    twist.angular.z = 0
    cmd_vel_pub.publish(twist)


if __name__ == '__main__':

    flag_task = Value('i', 1)  # 任务开启标志位
    cmd_chassis = Array('d', 4)  # 底盘控制
    slow_down = Value('i', 0)
    on_position = Value('i', 0)
    cmd_chassis[0] = arm  # 解锁

    p_chassis = Process(target=chassis_control, args=(flag_task, cmd_chassis, slow_down, on_position))  # 底盘控制子进程

    print("\n开启底盘控制进程\n")
    p_chassis.start()
    time.sleep(.5)
    
    while flag_task.value:
        print("\n代号      指令")
        print("se        启动引擎")
        print("cl        校准线速度")
        print("ca        校准角速度")
        print("ke        关闭引擎")
        print("q         退出")
        command = input("\n请输入指令代码: \n>>>")
        if command == "se":
            print("\n当前指令  启动引擎\n")
            cmd_chassis[0] = arm
            time.sleep(1)
        elif command == "cl":
            print("\n当前指令  校准线速度\n")
            drive_calib('linear')
        elif command == "ca":
            print("\n当前指令  校准角速度\n")
            drive_calib('angular')
        elif command == "ke":
            print("\n当前指令  关闭引擎\n")
            cmd_chassis[0] = disarm
            time.sleep(1)
        elif command == "q":
            print("\n当前指令  退出\n")
            break
        else:
            print("\n未知指令 请重新输入\n")


    flag_task.value = 0

    p_chassis.join()
