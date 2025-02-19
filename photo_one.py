import cv2
import time

# 拍摄人识别名称
# 日期
# 保存路径
paisheren = "bulletproof_vest"
riqi = "4-18"
# 指定保存路径
save_path = "/home/ucar/visial_line/bulletproof_vest/"

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 设置分辨率，例如设置宽度为640，高度为480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 设置帧率，例如设置为30帧每秒
cap.set(cv2.CAP_PROP_FPS, 30)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

count = 0

# 实时获取摄像头获取的图像
while True:
    # 读取当前帧
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # 显示图像
    cv2.imshow('Video Cam', frame)

    # 每当用户按下空格键（ASCII码为32），就保存当前帧为图片
    key = cv2.waitKey(1)
    if key == ord(' '):  # 按下空格键
        image_path = save_path + str(paisheren) + str(riqi) + str("_") + str(count) + str(".jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved image: {image_path}")
        count += 1

# 用户按下'q'键，退出循环
    if key == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()  # 关闭所有的窗口
