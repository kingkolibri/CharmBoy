#!/usr/bin/env python

import sys
import rospy
from std_msgs.msg import String
from roboy_cognition_msgs.srv import Talk
from charmboy.msg import Compliment


def callback(compliment):
    mes = compliment.phrase
    print(mes)
    rospy.wait_for_service('/roboy/cognition/speech/synthesis/talk')
    try:
        stt = rospy.ServiceProxy('/roboy/cognition/speech/synthesis/talk', Talk)
        resp = stt(compliment.phrase)
        print(resp.success)
    except rospy.ServiceException as e:
        print("Service call failed: %s", e)


def listener():
    rospy.init_node('roboy_talker', anonymous=True)
    rospy.Subscriber('roboy_compliments', Compliment, callback)

    rospy.spin()


if __name__ == "__main__":
    listener()
