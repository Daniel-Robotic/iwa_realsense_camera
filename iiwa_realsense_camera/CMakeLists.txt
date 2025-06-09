cmake_minimum_required(VERSION 3.20)
project(iiwa_realsense_camera LANGUAGES CXX)

add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(nlohmann-json_IMPLICIT_CONVERSIONS OFF)

# ─── Зависимости ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(iiwa_realsense_interfaces REQUIRED)
find_package(realsense2 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# GCC < 10: нужно std::filesystem вручную
set(EXTRA_LINK_LIBS "")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
  list(APPEND EXTRA_LINK_LIBS stdc++fs)
endif()

# ─── Библиотека‑обёртка RealSense (hpp/ cpp) ─────────────────────────────────
add_library(${PROJECT_NAME}_lib
  src/realsense_camera_wrapper.cpp)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}_lib PRIVATE
  realsense2::realsense2
  nlohmann_json::nlohmann_json
  ${EXTRA_LINK_LIBS})

# ─── Основной узел ROS2 ───────────────────────────────────────────────────────
add_executable(camera_handler src/camera_handler.cpp)

target_include_directories(camera_handler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(camera_handler
  ${PROJECT_NAME}_lib
  ${OpenCV_LIBRARIES}
  pybind11::embed
  ${EXTRA_LINK_LIBS})

ament_target_dependencies(camera_handler
  rclcpp OpenCV nlohmann_json std_srvs sensor_msgs
  iiwa_realsense_interfaces realsense2 ament_index_cpp)

# ─── Установка ────────────────────────────────────────────────────────────────
install(TARGETS camera_handler
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include)

ament_python_install_package(${PROJECT_NAME}
        PACKAGE_DIR python)

install(DIRECTORY models
        DESTINATION share/${PROJECT_NAME})

ament_package()
