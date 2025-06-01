cmake_minimum_required(VERSION 3.8)
project(iiwa_realsense_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)


set(msg_files
  msg/Detections.msg
)

set(srv_files
  srv/ChangeProfile.srv
  srv/CropImage.srv
  srv/IntelCameraInformation.srv
  srv/GetDepthAtPoint.srv
  srv/Get3DPoint.srv
  srv/ChangeModel.srv
  srv/ListModels.srv
)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  ${msg_files}
  ${srv_files}

  DEPENDENCIES std_msgs
)
ament_export_dependencies(rosidl_default_runtime std_msgs)

ament_package()
