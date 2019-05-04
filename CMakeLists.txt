cmake_minimum_required(VERSION 2.8.3)
project(charmboy)

find_package(catkin REQUIRED COMPONENTS dynamic_reconfigure message_generation roscpp rosgraph_msgs rospy std_msgs)

add_message_files(
  FILES
  CharmboyMsg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS dynamic_reconfigure message_runtime roscpp rosgraph_msgs rospy std_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

install(PROGRAMS src/pylistener.py src/pytalker.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
