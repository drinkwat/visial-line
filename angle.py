import rospy
import time
from geometry_msgs.msg import Twist




def chassis_control():
    """
        底盘控制进程

        Args:
            flag_task_c:        巡线任务开启标志位 (共享)  1 进行巡线 0 结束巡线
            cmd_chassis_c:      底盘控制 (共享) 四元数，控制底盘速度
            with_chassis_c:     开启底盘

        Returns:
            None
    """
    
    
    rospy.init_node('linetrack', anonymous=True)
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.loginfo("这里是行驶进程")  

    msg = Twist()
    turn_left = 1
    
    while True:
        


            
            
        rate = rospy.Rate(10)
        msg.linear.x = 0.4
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0

        if turn_left:
            msg.angular.z = 2.5
            
        else:
            msg.angular.z = -2.6


        rospy.loginfo("开始旋转")

        for i in range(5):
            cmd_vel_pub.publish(msg)
            time.sleep(0.1)

        rospy.loginfo("旋转结束")

        time.sleep(0.5)

            
        while True:
            
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0; 
            cmd_vel_pub.publish(msg)
                
        time.sleep(0.1)
            

        

    rospy.loginfo("行驶进程结束")


if __name__ == "__main__":
   chassis_control()