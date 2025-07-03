#!/usr/bin/env python3
"""
Простой launch-файл: запускает camera_handler и подсовывает JSON-файл с параметрами.

Запуск:
    ros2 launch iiwa_realsense_bringup camera_handler_launch.py
    ros2 launch iiwa_realsense_bringup camera_handler_launch.py \
        param_file:=/path/to/other.json  node_name:=front_cam
"""

import json
from pathlib import Path, PurePath
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_path
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def default_json():
    share = get_package_share_path("iiwa_realsense_bringup")
    return str(PurePath(share, "config", "setting.json"))


def make_node(context, *_):
    cfg_path = Path(LaunchConfiguration("param_file").perform(context)).expanduser()
    params = json.load(cfg_path.open())

    node_name = LaunchConfiguration("node_name").perform(context) or params.pop("node_name", "camera_handler_node")

    # Добавляем в параметры полный путь к конфигу как строку
    params["config_file"] = str(cfg_path)

    return [
        Node(
            package="iiwa_realsense_camera",
            executable="camera_handler",
            name=node_name,
            parameters=[params],
            output="screen"
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("param_file",
                              default_value=default_json(),
                              description="JSON with parameters"),
        DeclareLaunchArgument("node_name",
                              default_value="",
                              description="override node_name"),
        OpaqueFunction(function=make_node)
    ])
