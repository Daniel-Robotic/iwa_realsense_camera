// This ROS2 node captures synchronized RGB and Depth frames from a RealSense camera,
// optionally applies zoom and crop, passes the RGB image to a Python model for inference
// (e.g., YOLOv8 with pose/segmentation/detection), and publishes both raw image and
// detection results to ROS topics. It also provides various services for runtime control.

// Includes for C++ standard and external libraries
#include <chrono>
#include <memory>
#include <string>
#include <Python.h>
#include <filesystem>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv4/opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

// ROS2 includes
#include "rclcpp/rclcpp.hpp" 
#include "std_srvs/srv/trigger.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

// Custom interface and service messages
#include "iiwa_realsense_interfaces/srv/intel_camera_information.hpp"
#include "iiwa_realsense_interfaces/srv/get_depth_at_point.hpp"
#include "iiwa_realsense_camera/realsense_camera_wrapper.hpp"
#include "iiwa_realsense_interfaces/srv/change_profile.hpp"
#include "iiwa_realsense_interfaces/srv/get3_d_point.hpp"
#include "iiwa_realsense_interfaces/srv/change_model.hpp"
#include "iiwa_realsense_interfaces/srv/list_models.hpp"
#include "iiwa_realsense_interfaces/msg/detections.hpp"
#include "iiwa_realsense_interfaces/srv/crop_image.hpp" 

// Namespace aliasing
using namespace std;
namespace py = pybind11;
using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono_literals;
using namespace pybind11::literals;
  

class CameraHandler : public rclcpp::Node {
  public:
    // Constructor:
    // - Declares and retrieves ROS parameters
    // - Initializes Python interpreter and loads the default model
    // - Creates camera wrapper and ROS publishers/services
    // - Starts periodic timer for image capture and processing
    CameraHandler() : Node("camera_handler_node")
     {
      this->declare_parameter("camera_name", "default_camera");
      this->declare_parameter("camera_width", 640);
      this->declare_parameter("camera_height", 480);
      this->declare_parameter("camera_fps", 30);
      this->declare_parameter("zoom_level", 1.0);
      this->declare_parameter("crop_x_offset", 0);
      this->declare_parameter("crop_y_offset", 0);
      this->declare_parameter("publish_image", true);
      this->declare_parameter("publish_detect_object", false);
      this->declare_parameter("publish_pointcloud", false);
      this->declare_parameter("model_name", "yolov8n-pose");
      this->declare_parameter("model_format", "pt");
      this->declare_parameter("model_conf", 0.3);
      this->declare_parameter("model_device", "cpu");
      this->declare_parameter("max_object_detection", 1);
      this->declare_parameter("model_task", "pose");
      this->declare_parameter("config_file", "./setting.json");


      camera_name_ = this->get_parameter("camera_name").as_string();
      width_ = static_cast<uint16_t>(this->get_parameter("camera_width").as_int());
      height_ = static_cast<uint16_t>(this->get_parameter("camera_height").as_int());
      fps_ = static_cast<uint16_t>(this->get_parameter("camera_fps").as_int());
      crop_x_offset_ = static_cast<uint16_t>(this->get_parameter("crop_x_offset").as_int());
      crop_y_offset_ = static_cast<uint16_t>(this->get_parameter("crop_y_offset").as_int());
      publish_image_ = static_cast<bool>(this->get_parameter("publish_image").as_bool());
      publish_detect_object_ = static_cast<bool>(this->get_parameter("publish_detect_object").as_bool());
      publish_pointcloud_ = static_cast<bool>(this->get_parameter("publish_pointcloud").as_bool());
      model_device_ = static_cast<string>(this->get_parameter("model_device").as_string());
      model_conf_ = static_cast<float>(this->get_parameter("model_conf").as_double());
      max_object_detection_= static_cast<uint8_t>(this->get_parameter("max_object_detection").as_int());
      config_file_ = this->get_parameter("config_file").as_string();
       

      auto zoom_param = this->get_parameter("zoom_level");
      zoom_level_ = zoom_param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE
                    ? static_cast<float>(zoom_param.as_double())
                    : static_cast<float>(zoom_param.as_int());


      // Realsense wrapper
      camera_ = std::make_shared<RealsenseCameraWrapper>(width_, height_, fps_);

      // Python init
      py::initialize_interpreter();
      try {
          py::module detect_mod = py::module::import("iiwa_realsense_camera.detect");

          load_model_func_ = detect_mod.attr("load_model");
          detect_func_ = detect_mod.attr("detect");
          
          RCLCPP_INFO(get_logger(), "detect.py imported successfully");

          py::dict out = load_model_func_(this->get_parameter("model_task").as_string(),
                                          this->get_parameter("model_name").as_string(), 
                                          this->get_parameter("model_format").as_string()).cast<py::dict>();
          RCLCPP_INFO(this->get_logger(), static_cast<string>(out["message"].cast<py::str>()).c_str());

      } catch (const py::error_already_set& e) {
          RCLCPP_FATAL(get_logger(), "Python error: %s", e.what());
          throw;
      }

      // ROS I/O
      publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
        "camera/" + camera_name_ + "/rgb/raw", 10);
      
      camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "camera/" + camera_name_ + "/camera_info", 10);

      detection_publisher_ = this->create_publisher<iiwa_realsense_interfaces::msg::Detections>(
        "camera/" + camera_name_ + "/rgb/detection", 10);
      
      pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        camera_name_ + "/pointcloud", rclcpp::SensorDataQoS());

      streaming_camera_service_ = this->create_service<std_srvs::srv::Trigger>(
        "camera/" + camera_name_ + "/streaming_camera",
        bind(&CameraHandler::handle_stop_streaming_camera, this, std::placeholders::_1, std::placeholders::_2));
      
      nn_detection_service_ = this->create_service<std_srvs::srv::Trigger>(
        "camera/" + camera_name_ + "/nn_detection",
        bind(&CameraHandler::handle_nn_detection, this, std::placeholders::_1, std::placeholders::_2));

      pointcloud_toggle_service_ = this->create_service<std_srvs::srv::Trigger>(
        "camera/" + camera_name_ + "/pointcloud_toggle",
        bind(&CameraHandler::handle_pointcloud_toggle, this, std::placeholders::_1, std::placeholders::_2));

      load_model_service_ = this->create_service<iiwa_realsense_interfaces::srv::ChangeModel>(
        "camera/" + camera_name_ + "/load_model",
        bind(&CameraHandler::handle_load_model, this, std::placeholders::_1, std::placeholders::_2));

      change_profile_service_ = this->create_service<iiwa_realsense_interfaces::srv::ChangeProfile>(
        "camera/" + camera_name_ + "/change_profile",
        bind(&CameraHandler::handle_change_profile, this, std::placeholders::_1, std::placeholders::_2));

      camera_info_service_ = this->create_service<iiwa_realsense_interfaces::srv::IntelCameraInformation>(
        "camera/" + camera_name_ + "/camera_information",
        bind(&CameraHandler::handle_info, this, std::placeholders::_1, std::placeholders::_2));

      crop_image_service_ = this->create_service<iiwa_realsense_interfaces::srv::CropImage>(
        "camera/" + camera_name_ + "/crop_image",
        bind(&CameraHandler::handle_crop_image, this, std::placeholders::_1, std::placeholders::_2));

      get_depth_service_ = this->create_service<iiwa_realsense_interfaces::srv::GetDepthAtPoint>(
        "camera/" + camera_name_ + "/get_depth_at_point",
        bind(&CameraHandler::handle_get_depth_at_point, this, std::placeholders::_1, std::placeholders::_2));

      get_3d_point_service_ = this->create_service<iiwa_realsense_interfaces::srv::Get3DPoint>(
        "camera/" + camera_name_ + "/get_3d_point",
        bind(&CameraHandler::handle_get_3d_point, this, std::placeholders::_1, std::placeholders::_2));

      list_models_service_ = create_service<iiwa_realsense_interfaces::srv::ListModels>(
          "camera/" + camera_name_ + "/list_models",
          bind(&CameraHandler::handle_list_models, this, std::placeholders::_1, std::placeholders::_2));

      save_image_service_ = this->create_service<std_srvs::srv::Trigger>(
          "camera/" + camera_name_ + "/save_image",
          bind(&CameraHandler::handle_save_image, this, std::placeholders::_1, std::placeholders::_2));
          
      clear_images_service_ = this->create_service<std_srvs::srv::Trigger>(
          "camera/" + camera_name_ + "/clear_images",
          bind(&CameraHandler::handle_clear_images, this, std::placeholders::_1, std::placeholders::_2));
          
      timer_ = this->create_wall_timer(
        chrono::milliseconds(1000/fps_),
        bind(&CameraHandler::timer_callback, this)
      );

      RCLCPP_INFO(this->get_logger(), "Camera %s has been started", camera_name_.c_str());
    }

    // Destructor:
    // - Finalizes Python interpreter
    // - Stops RealSense camera streaming
    ~CameraHandler() {
      py::finalize_interpreter();
      camera_->stopStreaming();
    }

  private:

    /**
     * @brief Main periodic timer callback.
     * 
     * - Captures aligned RGB and depth frames from the camera.
     * - Builds and publishes CameraInfo message.
     * - Optionally:
     *    - Applies zoom and cropping to color image.
     *    - Runs neural network inference and publishes detection results.
     *    - Publishes compressed image.
     *    - Publishes point cloud.
     */
    void timer_callback() {
        auto [depth_frame, color_frame, intrinsics] = camera_->getAlignedImages();
        depth_image_ = depth_frame;
        color_image_ = color_frame;
        rclcpp::Time ros_time = now();

        auto info_msg = create_camera_info_msg(intrinsics, ros_time);
        apply_zoom_if_needed();

        if (publish_detect_object_) publish_detections(color_image_, ros_time);
        if (publish_image_) publish_compressed_image(ros_time);
        if (publish_pointcloud_) publish_pointcloud_msg(ros_time);

        camera_info_publisher_->publish(info_msg);
    }

    /**
     * @brief Constructs a CameraInfo message from RealSense intrinsics.
     * 
     * @param intrinsics RealSense camera intrinsics (fx, fy, ppx, ppy, distortion).
     * @param stamp Timestamp to assign to the message.
     * @return sensor_msgs::msg::CameraInfo Filled camera info message.
     */
    sensor_msgs::msg::CameraInfo create_camera_info_msg(const rs2_intrinsics& intrinsics, const rclcpp::Time& stamp) {
        sensor_msgs::msg::CameraInfo msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = camera_name_;
        msg.height = intrinsics.height;
        msg.width = intrinsics.width;
        msg.distortion_model = "plumb_bob";

        msg.d.resize(5, 0.0);
        for (int i = 0; i < 5; ++i) msg.d[i] = intrinsics.coeffs[i];

        msg.k = {intrinsics.fx, 0.0, intrinsics.ppx,
                0.0, intrinsics.fy, intrinsics.ppy,
                0.0, 0.0, 1.0};

        msg.r = {1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};

        msg.p = {intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
                0.0, intrinsics.fy, intrinsics.ppy, 0.0,
                0.0, 0.0, 1.0, 0.0};

        return msg;
    }

    /**
     * @brief Applies digital zoom and optional cropping to the color image.
     * 
     * If `zoom_level_ != 1.0`, crops the image around a center point (or offset) and
     * resizes it back to the original resolution.
     */
    void apply_zoom_if_needed() {
        if (zoom_level_ == 1.0f) return;

        int width = color_image_.cols;
        int height = color_image_.rows;
        int new_width = static_cast<int>(width / zoom_level_);
        int new_height = static_cast<int>(height / zoom_level_);
        int x1 = crop_x_offset_;
        int y1 = crop_y_offset_;
        int x2 = x1 + new_width;
        int y2 = y1 + new_height;

        if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
            x1 = (width - new_width) / 2;
            y1 = (height - new_height) / 2;
        }

        cv::Rect roi(x1, y1, new_width, new_height);
        color_image_ = color_image_(roi);
        cv::resize(color_image_, color_image_, cv::Size(width, height));
    }

    /**
     * @brief Performs object detection or pose/segment inference on the given RGB image.
     * 
     * - Converts the RGB OpenCV image into a NumPy array.
     * - Runs Python detection function (`detect_func_`) via pybind11.
     * - Parses results: bounding boxes, keypoints, masks, OBBs.
     * - Draws results over the image.
     * - Publishes detection results as `iiwa_realsense_interfaces::msg::Detections`.
     * 
     * @param rgb Input/output image to run detection on and annotate in-place.
     * @param stamp ROS time stamp for message headers.
     */
    void publish_detections(cv::Mat& rgb, const rclcpp::Time& stamp) {
        if (rgb.empty()) return;
        if (!rgb.isContinuous()) rgb = rgb.clone();

        const std::vector<ssize_t> shape   = { rgb.rows, rgb.cols, 3 };
        const std::vector<ssize_t> strides = { static_cast<ssize_t>(rgb.step[0]),
                                              static_cast<ssize_t>(rgb.step[1]),
                                              1 };

        py::gil_scoped_acquire gil;
        py::array_t<uint8_t> np_img(shape, strides, rgb.data, py::none());

        int max_arg = (max_object_detection_ == 0) ? -1 : max_object_detection_;
        py::dict out = detect_func_(np_img,
                                    "conf"_a = model_conf_,
                                    "device"_a = model_device_,
                                    "max_objects"_a = max_arg)
                      .cast<py::dict>();

        std::string task = out["task"].cast<std::string>();
        auto boxes    = out["boxes"].cast<py::array_t<float>>();
        auto scores   = out["scores"].cast<py::array_t<float>>();
        auto class_id = out["class_id"].cast<py::array_t<int>>();
        auto kps_obj  = out["keypoints"];
        auto masks_obj= out["masks"];
        auto obb_obj  = out["obb"];

        // ----------- Drawing -----------
        if ((task == "detect" || task == "pose" || task == "segment") && boxes.ndim() == 2 && boxes.shape(1) == 4) {
            auto b = boxes.unchecked<2>();
            for (ssize_t i = 0; i < boxes.shape(0); ++i) {
                cv::rectangle(rgb,
                              { int(b(i,0)), int(b(i,1)) },
                              { int(b(i,2)), int(b(i,3)) },
                              { 0, 255, 0 }, 2);
            }
        }

        if (task == "pose" && py::isinstance<py::array>(kps_obj)) {
            auto kps = kps_obj.cast<py::array_t<float>>();
            if (kps.ndim() == 3 && kps.shape(2) >= 2) {
                auto kp = kps.unchecked<3>();
                for (ssize_t n = 0; n < kps.shape(0); ++n) {
                    for (int i = 0; i < std::min<int>(17, kps.shape(1)); ++i) {
                        int x = static_cast<int>(kp(n,i,0));
                        int y = static_cast<int>(kp(n,i,1));
                        if (x == 0 && y == 0) continue;
                        cv::circle(rgb, {x, y}, 3, {0, 255, 255}, -1);
                    }
                    for (const auto& [a, b] : COCO_PAIRS_) {
                        if ((kp(n,a,0)||kp(n,a,1)) && (kp(n,b,0)||kp(n,b,1))) {
                            cv::line(rgb,
                                    {int(kp(n,a,0)), int(kp(n,a,1))},
                                    {int(kp(n,b,0)), int(kp(n,b,1))},
                                    {0, 255, 0}, 2);
                        }
                    }
                }
            }
        }

        if (task == "segment" && py::isinstance<py::array>(masks_obj)) {
            auto masks = masks_obj.cast<py::array_t<bool>>();
            if (masks.ndim() == 3) {
                auto m = masks.unchecked<3>();
                for (ssize_t n = 0; n < masks.shape(0); ++n) {
                    cv::Mat mask(rgb.rows, rgb.cols, CV_8U, (void*)m.data(n,0,0));
                    cv::Mat colored;
                    cv::cvtColor(mask * 255, colored, cv::COLOR_GRAY2BGR);
                    cv::addWeighted(rgb, 1.0, colored, 0.5, 0, rgb);
                }
            }
        }

        if (task == "obb" && py::isinstance<py::array>(obb_obj)) {
            auto obb = obb_obj.cast<py::array_t<float>>();
            if (obb.ndim() == 2 && obb.shape(1) == 5) {
                auto o = obb.unchecked<2>();
                for (ssize_t i = 0; i < obb.shape(0); ++i) {
                    cv::RotatedRect rr({o(i,0), o(i,1)}, {o(i,2), o(i,3)}, o(i,4) * 180 / M_PI);
                    cv::Point2f pts[4]; rr.points(pts);
                    for (int j = 0; j < 4; ++j) {
                        cv::line(rgb, pts[j], pts[(j+1)%4], {0,255,0}, 2);
                    }
                }
            }
        }

        // ----------- Fill Detections message -----------
        iiwa_realsense_interfaces::msg::Detections detection_msg_;
        detection_msg_.header.stamp = stamp;
        detection_msg_.header.frame_id = camera_name_;
        detection_msg_.task = task;

        const size_t N = boxes.shape(0);
        detection_msg_.num_detections = static_cast<uint32_t>(N);

        detection_msg_.boxes_xyxy.assign(boxes.data(), boxes.data() + boxes.size());
        detection_msg_.scores.assign(scores.data(), scores.data() + scores.size());
        detection_msg_.class_id.assign(class_id.data(), class_id.data() + class_id.size());

        if (py::isinstance<py::array>(kps_obj)) {
            auto kps = kps_obj.cast<py::array_t<float>>();
            detection_msg_.keypoints_xy.assign(kps.data(), kps.data() + kps.size());
        } else {
            detection_msg_.keypoints_xy.clear();
        }

        if (py::isinstance<py::array>(masks_obj)) {
            auto buf = masks_obj.cast<py::array>();
            py::buffer_info info = buf.request();
            detection_msg_.masks.resize(info.size * info.itemsize);
            std::memcpy(detection_msg_.masks.data(), info.ptr, detection_msg_.masks.size());
        } else {
            detection_msg_.masks.clear();
        }

        if (py::isinstance<py::array>(obb_obj)) {
            auto obb = obb_obj.cast<py::array_t<float>>();
            detection_msg_.obb_xywhr.assign(obb.data(), obb.data() + obb.size());
        } else {
            detection_msg_.obb_xywhr.clear();
        }

        detection_publisher_->publish(detection_msg_);
    }

    /**
     * @brief Publishes the current color image as a JPEG-compressed ROS image.
     * 
     * - Encodes the image using OpenCV to JPEG format.
     * - Wraps it into a `sensor_msgs::msg::CompressedImage` and publishes it.
     * 
     * @param stamp ROS time stamp for message header.
     */
    void publish_compressed_image(const rclcpp::Time& stamp) {
        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = camera_name_;
        msg.format = "jpeg";

        std::vector<uchar> buf;
        cv::imencode(".jpg", color_image_, buf);
        msg.data = std::move(buf);

        publisher_->publish(msg);
    }

    /**
     * @brief Generates and publishes a 3D point cloud from aligned depth and color frames.
     * 
     * - Uses RealSense SDK to generate point cloud from depth + color frames.
     * - Constructs a `sensor_msgs::msg::PointCloud2` message with x/y/z fields.
     * - Publishes the point cloud message.
     * 
     * @param stamp ROS time stamp for message header.
     */
    void publish_pointcloud_msg(const rclcpp::Time& stamp) {
        rs2::frameset frames = camera_->getLatestFrameset();
        auto depth_f = frames.get_depth_frame();
        auto color_f = frames.get_color_frame();

        auto [points, _] = camera_->getPointCloud(depth_f, color_f);

        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = stamp;
        cloud_msg.header.frame_id = camera_name_;
        cloud_msg.height = 1;
        cloud_msg.width = static_cast<uint32_t>(points.size());
        cloud_msg.is_dense = false;
        cloud_msg.is_bigendian = false;
        cloud_msg.point_step = 3 * sizeof(float);
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

        cloud_msg.fields.resize(3);
        const char* names[3] = {"x", "y", "z"};
        for (int i = 0; i < 3; ++i) {
            cloud_msg.fields[i].name = names[i];
            cloud_msg.fields[i].offset = i * sizeof(float);
            cloud_msg.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
            cloud_msg.fields[i].count = 1;
        }

        cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);
        uint8_t* ptr = cloud_msg.data.data();

        for (const auto& p : points) {
            memcpy(ptr, p.data(), 3 * sizeof(float));
            ptr += 3 * sizeof(float);
        }

        pointcloud_pub_->publish(cloud_msg);
    }

    // Service callback:
    // - Saves the current RGB frame to the "images" folder
    // - The filename format is <index>_<hhmm>_<ddmmyyyy>.jpg (e.g., 1_1430_09062025.jpg)
    // - Creates the folder if it does not exist
    // - Increments the index on each save
    // - Returns a success flag and message
    void handle_save_image([[maybe_unused]] const shared_ptr<std_srvs::srv::Trigger::Request> request,
      shared_ptr<std_srvs::srv::Trigger::Response> response) {
      try {
        if (color_image_.empty()) {
          response->success = false;
          response->message = "No image available to save.";
          return;
        }

        std::string folder = "images";
        if (!fs::exists(folder)) fs::create_directories(folder);

        // Count existing files
        size_t image_count = std::count_if(fs::directory_iterator(folder), fs::directory_iterator(), [](const auto& entry) {
          return entry.is_regular_file() && entry.path().extension() == ".jpg";
        });

        // Get time
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&t);

        // Create filename
        char filename[256];
        std::snprintf(filename, sizeof(filename), "%zu_%02d%02d_%02d%02d%04d.jpg", 
        image_count + 1, tm.tm_hour, tm.tm_min, tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900);

        fs::path filepath = fs::path(folder) / filename;

        cv::imwrite(filepath.string(), color_image_);
        response->success = true;
        response->message = "Image saved: " + filepath.string();
        RCLCPP_INFO(this->get_logger(), response->message.c_str());
      } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
        RCLCPP_ERROR(this->get_logger(), e.what());
      }
    }

    // Service callback:
    // - Deletes all files from the "images" folder if it exists
    // - Skips deletion if the folder is missing
    // - Returns the number of deleted files and status message
    void handle_clear_images([[maybe_unused]] const shared_ptr<std_srvs::srv::Trigger::Request> request,
                              shared_ptr<std_srvs::srv::Trigger::Response> response) {
      try {
        std::string folder = "images";
        if (!fs::exists(folder)) {
          response->success = true;
          response->message = "Directory does not exist. Nothing to delete.";
          return;
        }
    
        size_t deleted = 0;
        for (const auto& entry : fs::directory_iterator(folder)) {
          if (entry.is_regular_file()) {
            fs::remove(entry);
            ++deleted;
          }
        }
    
        response->success = true;
        response->message = "Deleted " + std::to_string(deleted) + " files.";
        RCLCPP_INFO(this->get_logger(), response->message.c_str());
      } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
        RCLCPP_ERROR(this->get_logger(), e.what());
      }
    }

    // Service callback:
    // - Input: empty Trigger request
    // - Toggles the "publish_image_" flag to start/stop publishing images
    // - Returns a response message with new status
    void handle_stop_streaming_camera([[maybe_unused]] const shared_ptr<std_srvs::srv::Trigger::Request> request,
                                      shared_ptr<std_srvs::srv::Trigger::Response> response) {
        publish_image_ = !publish_image_;
        this->set_parameter(rclcpp::Parameter("publish_image", publish_image_));
        save_params_to_json();

        string msg = "Camera status: ";
        msg += publish_image_ ? "stream on" : "stream off";
        RCLCPP_INFO(this->get_logger(), msg.c_str());

        response->success = true;
        response->message = msg;
    }

    // Service callback:
    // - Input: empty Trigger request
    // - Toggles the "publish_detect_object_" flag to enable/disable detection
    // - Returns a response message with updated status
    void handle_nn_detection([[maybe_unused]] const shared_ptr<std_srvs::srv::Trigger::Request> request,
                              shared_ptr<std_srvs::srv::Trigger::Response> response) {
        publish_detect_object_ = !publish_detect_object_;
        this->set_parameter(rclcpp::Parameter("publish_detect_object", publish_detect_object_));
        save_params_to_json();

        string msg = "Neural network status: ";
        msg += publish_detect_object_ ? "on" : "off";
        RCLCPP_INFO(this->get_logger(), msg.c_str());

        response->success = true;
        response->message = msg;
    }

    void handle_pointcloud_toggle([[maybe_unused]] const shared_ptr<std_srvs::srv::Trigger::Request> request,
                              shared_ptr<std_srvs::srv::Trigger::Response> response) {
        publish_pointcloud_ = !publish_pointcloud_;
        this->set_parameter(rclcpp::Parameter("publish_pointcloud", publish_pointcloud_));
        save_params_to_json();
        
        string msg = "Pointcloud publish status: ";
        msg += publish_pointcloud_ ? "on" : "off";
        RCLCPP_INFO(this->get_logger(), msg.c_str());

        response->success = true;
        response->message = msg;
    }

    // Service callback:
    // - Lists all .pt/.engine models from the models directory
    // - Input: empty request
    // - Output: success flag, list of model paths, and message
    void handle_list_models([[maybe_unused]] const std::shared_ptr<iiwa_realsense_interfaces::srv::ListModels::Request> request,
      std::shared_ptr<iiwa_realsense_interfaces::srv::ListModels::Response> response) {
      try {
        auto pkg_root   = ament_index_cpp::get_package_share_directory("iiwa_realsense_camera");
        fs::path models_dir = fs::path(pkg_root) / "models";

        if (!fs::exists(models_dir)) {
          response->success = false;
          response->message = "models directory not found";
          RCLCPP_WARN(this->get_logger(), response->message.c_str());
          return;
        }

        for (const auto& entry : fs::recursive_directory_iterator(models_dir))
          if (entry.is_regular_file() && entry.path().extension() == ".pt" || entry.path().extension() == ".engine") {
            response->models.push_back(fs::relative(entry.path(), models_dir).string());
        }

        response->success = true;
        response->message = "OK";
      }
      catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
        RCLCPP_ERROR(this->get_logger(), response->message.c_str());
      }

    }

    // Service callback:
    // - Loads a new model for detection/pose/segmentation tasks
    // - Validates the request (confidence ∈ [0,1], valid device, etc.)
    // - Calls Python load_model() and stores the configuration
    // - Returns success status and load result message
    void handle_load_model(const shared_ptr<iiwa_realsense_interfaces::srv::ChangeModel::Request> request,
                           shared_ptr<iiwa_realsense_interfaces::srv::ChangeModel::Response> response) {
      
      
      if (request->confidence < 0.f || request->confidence > 1.f) {
        response->status = false;
        response->message = "confidence must be in [0,1]";
        RCLCPP_WARN(get_logger(), response->message.c_str());
        return;
      }

      string dev = request->device;
      transform(dev.begin(), dev.end(), dev.begin(), ::tolower);
      if (dev != "cpu" && dev != "cuda") {
          response->status = false;
          response->message = "device must be 'cpu' or 'cuda'";
          RCLCPP_WARN(get_logger(), response->message.c_str());
          return;
      }

      if (request->max_objects == 0) {
          response->status = false;
          response->message = "max_objects must be > 0";
          RCLCPP_WARN(get_logger(), response->message.c_str());
          return;
      }

      model_conf_ = request->confidence;
      model_device_ = dev;
      max_object_detection_ = request->max_objects;

      this->set_parameter(rclcpp::Parameter("model_conf", model_conf_));
      this->set_parameter(rclcpp::Parameter("model_device", model_device_));
      this->set_parameter(rclcpp::Parameter("max_object_detection", max_object_detection_));

      // Temporarily disable detection during model load
      bool publish_status = publish_detect_object_;
      publish_detect_object_ = false;
      this->set_parameter(rclcpp::Parameter("publish_detect_object", publish_detect_object_));
      save_params_to_json();

      py::dict out = load_model_func_(request->task, request->model_name, request->model_format).cast<py::dict>();
      bool status = out["status"].cast<py::bool_>();
      string message = out["message"].cast<py::str>();

      RCLCPP_INFO(this->get_logger(), message.c_str());

      response->status = status;
      response->message = message;

      // Restore previous detection publish state
      publish_detect_object_ = publish_status;
      this->set_parameter(rclcpp::Parameter("publish_detect_object", publish_detect_object_));
      save_params_to_json();
    }

    // Service callback:
    // - Changes resolution and FPS of the camera
    // - Invokes camera wrapper changeCameraProfile()
    // - Returns status and message based on success/failure
    void handle_change_profile(const shared_ptr<iiwa_realsense_interfaces::srv::ChangeProfile::Request> request,
                               shared_ptr<iiwa_realsense_interfaces::srv::ChangeProfile::Response> response) {
      bool status = camera_->changeCameraProfile(request->width, request->height, request->fps);
      response->status = status;
      response->message = status ? "Profile changed" : "Failed to change profile";

      if (status) {
          this->set_parameter(rclcpp::Parameter("camera_width", request->width));
          this->set_parameter(rclcpp::Parameter("camera_height", request->height));
          this->set_parameter(rclcpp::Parameter("camera_fps", request->fps));
          save_params_to_json();
          RCLCPP_INFO(this->get_logger(), "Profile changed");
      } else {
          RCLCPP_WARN(this->get_logger(), "Failed to change profile");
      }
    }

    // Service callback:
    // - Returns camera metadata including name, USB type, supported stream profiles, etc.
    // - Fills all fields of IntelCameraInformation.srv
    void handle_info([[maybe_unused]] const std::shared_ptr<iiwa_realsense_interfaces::srv::IntelCameraInformation::Request> request,
                     std::shared_ptr<iiwa_realsense_interfaces::srv::IntelCameraInformation::Response> response) {
      auto info = camera_->getCameraInformation();
      response->device_name = info.device_name;
      response->product_line = info.product_line;
      response->serial_number = info.serial_number;
      response->firmware_version = info.firmware_version;
      response->usb_type = info.usb_type;
      response->color_profile = info.color_profile;
      response->depth_profile = info.depth_profile;
    }

    // Service callback:
    // - Updates zoom level and cropping offset for the image stream
    // - Ensures values are positive and within bounds
    // - Updates internal parameters and returns result
    void handle_crop_image(const std::shared_ptr<iiwa_realsense_interfaces::srv::CropImage::Request> request,
                           std::shared_ptr<iiwa_realsense_interfaces::srv::CropImage::Response> response) {
      if (request->zoom_level <= 0 || request->crop_x_offset < 0 || request->crop_y_offset < 0) {
        string msg = "Invalid crop parameters";
        RCLCPP_WARN(this->get_logger(), msg.c_str());

        response->success = false;
        response->message = msg;
        return;
      }

      zoom_level_ = request->zoom_level;
      crop_x_offset_ = request->crop_x_offset;
      crop_y_offset_ = request->crop_y_offset;

      this->set_parameter(rclcpp::Parameter("zoom_level", zoom_level_));
      this->set_parameter(rclcpp::Parameter("crop_x_offset", crop_x_offset_));
      this->set_parameter(rclcpp::Parameter("crop_y_offset", crop_y_offset_));
      save_params_to_json();

      string msg = "Crop updated";
      RCLCPP_INFO(this->get_logger(), msg.c_str());

      response->success = true;
      response->message = msg;
    }


    // Service callback:
    // - Returns depth value at (x, y) pixel coordinate
    // - Validates bounds
    // - Reads depth from latest frame and returns it
    void handle_get_depth_at_point(
        const std::shared_ptr<iiwa_realsense_interfaces::srv::GetDepthAtPoint::Request> request,
        std::shared_ptr<iiwa_realsense_interfaces::srv::GetDepthAtPoint::Response> response)
    {
        try {
            auto frameset = camera_->getLatestFrameset();
            rs2::depth_frame depth_frame = frameset.get_depth_frame();

            uint16_t x = request->x;
            uint16_t y = request->y;

            if (!depth_frame || x >= depth_frame.get_width() || y >= depth_frame.get_height()) {
                response->success = false;
                response->message = "Coordinates out of bounds or depth frame unavailable";
                response->depth = 0.0f;
                return;
            }

            // Получаем дистанцию в метрах
            float depth_meters = depth_frame.get_distance(x, y);

            response->success = true;
            response->message = "Depth successfully obtained";
            response->depth = depth_meters;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in get_depth_at_point: %s", e.what());
            response->success = false;
            response->message = e.what();
            response->depth = 0.0f;
        }
    }


    // Service callback:
    // - Converts pixel (x, y) to 3D coordinates in camera space
    // - Uses RealSense intrinsics and deprojection function
    // - Returns (X, Y, Z) point or failure message
    void handle_get_3d_point(const std::shared_ptr<iiwa_realsense_interfaces::srv::Get3DPoint::Request> request,
                            std::shared_ptr<iiwa_realsense_interfaces::srv::Get3DPoint::Response> response) {
      try {
        uint16_t x = request->x;
        uint16_t y = request->y;

        string msg = "3D point obtained successfully";

        if (x >= depth_image_.cols || y >= depth_image_.rows) {
          msg = "Coordinates out of bounds";
          response->success = false;
          response->message = msg;
          return;
        }
  ;
        rs2::frameset frames = camera_->getLatestFrameset();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        float depth_m = depth_frame.get_distance(x, y);

        rs2::video_stream_profile depth_profile = depth_frame.get_profile().as<rs2::video_stream_profile>();
        rs2_intrinsics intrin = depth_profile.get_intrinsics();

        float pixel[2] = {static_cast<float>(x), static_cast<float>(y)};
        float point[3];
        rs2_deproject_pixel_to_point(point, &intrin, pixel, depth_m);

        RCLCPP_INFO(this->get_logger(), msg.c_str());
        response->x = point[0];
        response->y = point[1];
        response->z = point[2];
        response->success = true;
        response->message = msg;
      } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), e.what());
        response->success = false;
        response->message = e.what();
      }
    }

    void save_params_to_json() {
        try {
            nlohmann::json j;

            // Если файл существует, попробуем сначала загрузить, чтобы сохранить неизменённые поля
            std::ifstream infile(config_file_);
            if (infile.good()) {
                infile >> j;
                infile.close();
            }

            // Обновляем параметры
            j["publish_image"] = publish_image_;
            j["publish_detect_object"] = publish_detect_object_;
            j["publish_pointcloud"] = publish_pointcloud_;
            j["zoom_level"] = zoom_level_;
            j["crop_x_offset"] = crop_x_offset_;
            j["crop_y_offset"] = crop_y_offset_;
            j["model_conf"] = model_conf_;
            j["model_device"] = model_device_;
            j["max_object_detection"] = max_object_detection_;

            // Записываем обратно в файл
            std::ofstream outfile(config_file_);
            outfile << j.dump(4) << std::endl;
            outfile.close();

            RCLCPP_INFO(this->get_logger(), "Parameters saved to %s", config_file_.c_str());
        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save parameters: %s", e.what());
        }
    }


    // --- Internal parameters and member variables ---
    float zoom_level_;
    float model_conf_;
    string camera_name_;
    string config_file_;
    string model_device_;
    py::object detect_func_;
    py::object load_model_func_;
    uint8_t max_object_detection_;
    uint16_t width_, height_, fps_;
    string model_name, model_format;
    cv::Mat depth_image_, color_image_;
    uint16_t crop_x_offset_, crop_y_offset_;
    bool publish_image_, publish_detect_object_, publish_pointcloud_;

    // For connecting human keypoints
    const vector<pair<int,int>> COCO_PAIRS_ = {
        {0,1},{0,2},{1,3},{2,4},        // head-shoulders
        {5,6},{5,7},{7,9},{6,8},{8,10}, // arms
        {5,11},{6,12},{11,12},          // hips
        {11,13},{13,15},{12,14},{14,16} // legs
    };

    rclcpp::TimerBase::SharedPtr timer_;  // Main periodic timer
    shared_ptr<RealsenseCameraWrapper> camera_;  // Camera wrapper object

    // ROS publishers
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;
    rclcpp::Publisher<iiwa_realsense_interfaces::msg::Detections>::SharedPtr detection_publisher_;

    // ROS services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr nn_detection_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr pointcloud_toggle_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr streaming_camera_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_image_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_images_service_;

    rclcpp::Service<iiwa_realsense_interfaces::srv::CropImage>::SharedPtr crop_image_service_;
    rclcpp::Service<iiwa_realsense_interfaces::srv::ChangeModel>::SharedPtr load_model_service_;
    rclcpp::Service<iiwa_realsense_interfaces::srv::ListModels>::SharedPtr list_models_service_;
    rclcpp::Service<iiwa_realsense_interfaces::srv::Get3DPoint>::SharedPtr get_3d_point_service_;
    rclcpp::Service<iiwa_realsense_interfaces::srv::GetDepthAtPoint>::SharedPtr get_depth_service_;
    rclcpp::Service<iiwa_realsense_interfaces::srv::ChangeProfile>::SharedPtr change_profile_service_;
    rclcpp::Service<iiwa_realsense_interfaces::srv::IntelCameraInformation>::SharedPtr camera_info_service_;

};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(make_shared<CameraHandler>());
  rclcpp::shutdown();
  return 0;
}