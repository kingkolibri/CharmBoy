#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from roboy_cognition_msgs.srv import *
from charmboy.msg import Compliment


facial_expressions = {
    'shy': 'S',
    'money': 'E',
    'kiss': 'K',
    'look left': 'L',
    'look right': 'R',
    'blink': 'B',
    'smile blink': 'W',
    'tongue out': 'D',
    'happy': 'Q',
    'lucky': 'Y',
    'hearts': 'H',
    'pissed': 'N',
    'angry': 'A',
    'irritated': 'X',
    'hypno eyes': 'V',
    'coloured': 'U',
    'rolling eyes': 'I',
    'surprised': 'Z',
    'pirate': 'P',
}


def callback(compliment):
    # request the corresponding face service according to the message
    mes = compliment.phrase
    print(mes)
    rospy.wait_for_service('/roboy/cognition/face/emotion')
    try:
        fs = rospy.ServiceProxy('/roboy/cognition/face/emotion', Emotion)
        # the following messages are only temporary
        resp = fs(compliment.expression)
        print(resp.success)
    except rospy.ServiceException as e:
        print("Service call failed: %s", e)

def listener():
    rospy.init_node('roboy_face_caller', anonymous = True)
    rospy.Subscriber('roboy_compliments', String, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
