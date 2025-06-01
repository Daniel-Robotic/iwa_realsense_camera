# iiwa_realsense_camera

![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Docs](https://img.shields.io/badge/docs-passing-brightgreen)

**iiwa_realsense_camera** is a ROS 2-compatible module for working with Intel RealSense cameras in a collaborative environment with the KUKA iiwa robot. It includes:
- Image capture and publishing
- Object detection, segmentation, pose estimation, and classification via YOLOv8
- ROS 2 services and messages for interaction with the camera and neural models

## Table of Contents
- [Messages and Services](#messages)
- [Parameters](#parameters)
- [Getting Started](#getting-started)
- [Organizations and Grants](#organizations)
- [Supported ROS 2 Distributions](#distributions)
- [Example Launch](#launch-example)

## Messages and Services <a name="messages"></a>

| Type | Name | Description |
|------|------|-------------|
| msg  | Detection.msg | Detection result: class IDs, bounding boxes, confidence scores, keypoints, masks. |
| srv  | ChangeModel.srv | Change the current inference model. |
| srv  | ChangeProfile.srv | Change camera resolution and FPS. |
| srv  | CropImage.srv | Zoom and crop offsets for the image. |
| srv  | Get3DPoint.srv | Retrieve 3D coordinates from RGB pixel. |
| srv  | GetDepthAtPoint.srv | Get depth value from a specific pixel. |
| srv  | IntelCameraInformation.srv | Get RealSense camera metadata. |
| srv  | ListModels.srv | List available models in model directory. |

## Parameters <a name="parameters"></a>

| Parameter | Description |
|-----------|-------------|
| `node_name` | Name of the ROS2 node. |
| `camera_name` | Camera name used in topics and services. |
| `camera_width`, `camera_height`, `camera_fps` | Resolution and framerate. |
| `zoom_level` | Digital zoom level (1.0 = no zoom). |
| `crop_x_offset`, `crop_y_offset` | Image cropping offsets. |
| `publish_image` | Whether to publish image frames. |
| `publish_detect_object` | Whether to publish detection results. |
| `model_name` | Name of the YOLOv8 model to load. |
| `model_format` | File format of the model (e.g. `.pt`, `.engine`). |
| `model_task` | Task type: `detect`, `segment`, `classify`, `pose`, `obb`. |
| `model_conf` | Confidence threshold for detection. |
| `model_device` | Inference device: `cpu` or `cuda`. |
| `max_object_detection` | Maximum number of objects to detect. |

## Getting Started <a name="getting-started"></a>

### Recommended: Using Docker

1. Make sure ROS2 is installed (Foxy/Humble/Jazzy)
2. Build interface package locally:
```bash
colcon build --packages-select iiwa_realsense_interfaces
```
3. In `docker-compose.yml`, choose your target platform:
   - `Dockerfile.x64` for x86-64 systems
   - `Dockerfile.jetson` for NVIDIA Jetson
4. Launch with:
```bash
docker-compose up --build
```

### Manual Setup

```bash
sudo apt install python3-vcstool
mkdir -p /ros2_ws/src
cd /ros2_ws/src
git clone https://github.com/Daniel-Robotic/iwa_realsense_camera.git .
vcpkg install
cd ..
colcon build
```

To run the node:
```bash
ros2 launch iiwa_realsense_camera camera_handler_launch.py
```

## Supported ROS 2 Distributions <a name="distributions"></a>

| ROS 2 Distro | Status |
|--------------|--------|
| Foxy         | ✅ Tested |
| Humble       | ✅ Tested |
| Jazzy        | ⭐ Recommended |

## Example Launch <a name="launch-example"></a>

```bash
ros2 launch iiwa_realsense_camera camera_handler_launch.py
```
This will start the camera node with all detection services enabled.

## Organizations and Grants <a name="organizations"></a>

We would further like to acknowledge following supporters:

| Logo | Notes |
|:--:|:---|
| <img src="https://knastu.ru/media/template/logo_knastu_block.svg?v2018" alt="wellcome" width="150" align="left">  | Thanks to [Komsomolsk-on-Amur State University](https://knastu.ru/) for providing classroom facilities for the robotic complex. |
| <img src="https://knastu.ru/media/files/page_files/education/etf/noc_prppt/Logo_NOTs_m.jpg" width="150" align="left"> | Thanks to the Scientific and Educational Center "Industrial Robotics and Advanced Industrial Technologies" for providing computing power and IntelRealsense cameras. |
| <img src="https://rscf.ru/local/templates/rscf/static/img/logo.svg" width="150" align="left"> | The work was carried out within the framework of a grant from the Russian Science Foundation, namely, "Development and synthesis of promising multimodal adaptive algorithms and methods for controlling the behavior of collaborative robotic systems, taking into account emergency situations and extreme conditions in a non-deterministic environment" [#22-71-10093](https://rscf.ru/project/22-71-10093/) and "Development of intelligent algorithms for building collaborative robotic control systems operating in a non-deterministic environmen" [#25-21-00292](https://rscf.ru/project/25-21-00292/) |
---