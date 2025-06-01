# Message used for representing the results of object detection, pose estimation,
# segmentation, or oriented bounding box detection in a single image frame.

# This message is typically published after a neural network processes an RGB image,
# and is designed to be compatible with multiple tasks such as:
# - Object Detection (bounding boxes)
# - Instance Segmentation (pixel-wise masks)
# - Pose Estimation (human keypoints)
# - Classification
# - Oriented Bounding Boxes (OBB)

# Example usage:
# - Robot vision system detects humans using a pose model (task = "pose")
# - The robot gets 2D keypoints from keypoints_xy and can further compute 3D keypoints
# - Alternatively, for object detection (task = "detect"), it uses boxes_xyxy and class_id


std_msgs/Header header
# Timestamp and coordinate frame of the detection results.
# Typically matches the camera timestamp and frame_id.

string task
# Type of detection task performed. Can be one of:
# "detect"   - object detection (bounding boxes)
# "segment"  - instance segmentation (masks)
# "classify" - image classification
# "pose"     - human pose estimation (keypoints)
# "obb"      - oriented bounding boxes (with rotation)

uint32 num_detections
# Number of detected objects in the current image.

float32[] boxes_xyxy
# Detected bounding boxes in [x_min, y_min, x_max, y_max] format.
# Length = num_detections * 4.
# Used in "detect", "segment", or "pose" tasks.

float32[] obb_xywhr
# Oriented bounding boxes in [center_x, center_y, width, height, rotation] (in radians).
# Length = num_detections * 5.
# Used in "obb" task.

float32[] keypoints_xy
# Keypoints for pose estimation in format [x1, y1, x2, y2, ..., x17, y17] for each detection.
# Length = num_detections * 17 * 2.
# Used in "pose" task (e.g. for COCO skeleton with 17 keypoints per person).

uint8[] masks
# Flattened binary masks for each detection (instance segmentation).
# The shape is N x H x W (number of objects x height x width) flattened into a 1D byte array.
# Used in "segment" task.

int32[] class_id
# Class ID for each detected object, corresponding to the model's label map.
# Length = num_detections.

float32[] scores
# Confidence scores for each detection (between 0 and 1).
# Length = num_detections.
