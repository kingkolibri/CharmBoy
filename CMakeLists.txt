cmake_minimum_required(VERSION 2.8.3)
project(charmboy)

find_package(catkin REQUIRED COMPONENTS
    message_generation
    rosgraph_msgs
    rospy
    std_msgs
    roboy_control_msgs
    roboy_cognition_msgs
    roboy_face
)

add_message_files(
    FILES
    Compliment.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
)

catkin_install_python(PROGRAMS src/pytalker.py src/roboy_face.py src/roboy_talk.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
