#!/usr/bin/env python
import os
import rospy

from charmboy.msg import Compliment
from compliment_picker import ComplimentPicker


class RoboyWhisperer(object):
    """Node roboy whisperer."""

    def __init__(self):

        self.enable = True

       # self.compliment_picker = ComplimentPicker("/home/kingkolibri/10_catkin_ws/src/CharmBoy/data/compliment_database.csv")

        if self.enable:
            """Turn on publisher."""
            self.pub = rospy.Publisher('roboy_compliments',
                                       Compliment,
                                       queue_size=1
                                       )

        else:
            self.stop()

    def stop(self):
        """Turn off publisher."""
        self.pub.unregister()

    def make_compliment(self, phrase, expression):
        if not self.enable:
            return

        msg = Compliment()
        msg.phrase = phrase
        msg.expression = expression

        self.pub.publish(msg)


# Main function.
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('roboy_whisperer')

    rate = rospy.Rate(1)  # 1hz
    # Go to class functions that do all the heavy lifting.
    try:
        whisperer = RoboyWhisperer()

        while not rospy.is_shutdown():
            whisperer.make_compliment("I must be in a museum, because you truly are a work of art.", "happy")

        rate.sleep()
    except rospy.ROSInterruptException:
        pass

    # Allow ROS to go to all callbacks.
    rospy.spin()
