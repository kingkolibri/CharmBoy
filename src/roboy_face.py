#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from roboy_control_msgs.msg import Emotion

facial_expressions = {
    'Shy': 'S',
    'Money': 'E',
    'Kiss': 'K',
    'Look left': 'L',
    'Look right': 'R',
    'Blink': 'B',
    'Smile blink': 'W',
    'Tongue out': 'D',
    'Happy': 'Q',
    'Lucky': 'Y',
    'Hearts': 'H',
    'Pissed': 'N',
    'Angry': 'A',
    'Irritated': 'X',
    'Hypno eyes': 'V',
    'Coloured': 'U',
    'Rolling eyes': 'I',
    'Surprised': 'Z',
    'Pirate': 'P',
}


def callback(data):

    pub = rospy.Publisher('/roboy/cognition/face/emotion', Emotion, queue_size=1)
    msg = Emotion()
    msg.emotion = facial_expressions.get(data.expression, 'Y')
    pub.publish(msg)


def listener():

    rospy.init_node('roboy_face_expressions', anonymous=True)
    rospy.Subscriber('roboy_compliments', String, callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
