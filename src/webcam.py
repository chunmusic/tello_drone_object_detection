#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import CompressedImage
import av
import cv2
import numpy as np
import threading
import traceback
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty



class StandaloneVideoStream(object):
    def __init__(self):
        self.cond = threading.Condition()
        self.queue = []
        self.closed = False

    def read(self, size):
        self.cond.acquire()
        try:
            if len(self.queue) == 0 and not self.closed:
                self.cond.wait(2.0)
            data = bytes()
            while 0 < len(self.queue) and len(data) + len(self.queue[0]) < size:
                data = data + self.queue[0]
                del self.queue[0]
        finally:
            self.cond.release()
        return data

    def seek(self, offset, whence):
        return -1

    def close(self):
        self.cond.acquire()
        self.queue = []
        self.closed = True
        self.cond.notifyAll()
        self.cond.release()

    def add_frame(self, buf):
        self.cond.acquire()
        self.queue.append(buf)
        self.cond.notifyAll()
        self.cond.release()


stream = StandaloneVideoStream()


def callback(msg):
    #rospy.loginfo('frame: %d bytes' % len(msg.data))
    stream.add_frame(msg.data)



def main():
    rospy.init_node('h264_listener')
    rospy.Subscriber("/tello/image_raw/h264", CompressedImage, callback)
    container = av.open(stream)
    rospy.loginfo('main: opened')



    cv2.imshow('Frame', res)
    cv2.waitKey(1)




if __name__ == '__main__':
    
    rospy.init_node('h264_listener')


# TODO USE TWIST JOGGER FOR CONTROLLING DRONE MANUALLY 


    pub_takeoff = rospy.Publisher('/tello/takeoff',Empty,queue_size=1)

    pub_land = rospy.Publisher('/tello/land',Empty,queue_size=1)

    pub_vel = rospy.Publisher('/tello/cmd_vel',Twist,queue_size=1)

    takeoff_msg = Empty()
    cmd_vel = Twist()     

    try:

        cmd_vel.linear.z = 0.7
        pub_vel.publish(cmd_vel)
        
        rospy.sleep(3)

        cmd_vel.linear.z = 0
        pub_vel.publish(cmd_vel)

        main()

    except BaseException:
        traceback.print_exc()
    finally:
        pub_land.publish(Empty())

        stream.close()
        cv2.destroyAllWindows()