std_msgs/Header header
string task              # detect | segment | classify | pose | obb
uint32 num_detections

float32[] boxes_xyxy      # N*4, detect/pose/segment
float32[] obb_xywhr       # N*5, obb
float32[] keypoints_xy    # N*17*2, pose
uint8[] masks            # flatten N*H*W, segment (bool array as bytes)

int32[] class_id        # N
float32[] scores          # N