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

// ROS2 includes
#include "rclcpp/rclcpp.hpp" 
#include "std_srvs/srv/trigger.hpp"
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
namespace py = pybind11;
namespace fs = std::filesystem;
using namespace std;
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
      this->declare_parameter("model_name", "yolov8n-pose");
      this->declare_parameter("model_format", "pt");
      this->declare_parameter("model_conf", 0.3);
      this->declare_parameter("model_device", "cpu");
      this->declare_parameter("max_object_detection", 1);
      this->declare_parameter("model_task", "pose");


      camera_name_ = this->get_parameter("camera_name").as_string();
      width_ = static_cast<uint16_t>(this->get_parameter("camera_width").as_int());
      height_ = static_cast<uint16_t>(this->get_parameter("camera_height").as_int());
      fps_ = static_cast<uint16_t>(this->get_parameter("camera_fps").as_int());
      crop_x_offset_ = static_cast<uint16_t>(this->get_parameter("crop_x_offset").as_int());
      crop_y_offset_ = static_cast<uint16_t>(this->get_parameter("crop_y_offset").as_int());
      publish_image_ = static_cast<bool>(this->get_parameter("publish_image").as_bool());
      publish_detect_object_ = static_cast<bool>(this->get_parameter("publish_detect_object").as_bool());
      model_device_ = static_cast<string>(this->get_parameter("model_device").as_string());
      model_conf_ = static_cast<float>(this->get_parameter("model_conf").as_double());
      max_object_detection_= static_cast<uint8_t>(this->get_parameter("max_object_detection").as_int());

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
      
      detection_publisher_ = this->create_publisher<iiwa_realsense_interfaces::msg::Detections>(
        "camera/" + camera_name_ + "/rgb/detection", 10);

      streaming_camera_service_ = this->create_service<std_srvs::srv::Trigger>(
        "camera/" + camera_name_ + "/streaming_camera",
        bind(&CameraHandler::handle_stop_streaming_camera, this, std::placeholders::_1, std::placeholders::_2));
      
      nn_detection_service_ = this->create_service<std_srvs::srv::Trigger>(
        "camera/" + camera_name_ + "/nn_detection",
        bind(&CameraHandler::handle_nn_detection, this, std::placeholders::_1, std::placeholders::_2));

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

    // Main timer callback:
    // - Captures aligned RGB and depth images
    // - Applies crop and zoom if requested
    // - Optionally runs neural network inference and draws results
    // - Publishes detection results and/or compressed image
    void timer_callback() {
      auto [depth_frame, color_frame] = camera_->getAlignedImages();

      depth_image_ = depth_frame;
      color_image_ = color_frame;
      rclcpp::Time ros_time = now();

      if (zoom_level_ != 1.0f) {
        int width = static_cast<int>(color_image_.cols);
        int height = static_cast<int>(color_image_.rows);
        int new_width = static_cast<int>(width / zoom_level_);
        int new_height = static_cast<int>(height / zoom_level_);
        int x1 = crop_x_offset_;
        int y1 = crop_y_offset_;
        int x2 = crop_x_offset_ + new_width;
        int y2 = crop_y_offset_ + new_height;

        if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
          x1 = (width - new_width) / 2;
          y1 = (height - new_height) / 2;
          x2 = x1 + new_width;
          y2 = y1 + new_height;
        }

        cv::Rect roi(x1, y1, new_width, new_height);
        color_image_ = color_image_(roi);
        cv::resize(color_image_, color_image_, cv::Size(width, height));
      }

      if (publish_detect_object_) {
        cv::Mat &rgb = color_image_;
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
        auto boxes = out["boxes"].cast<py::array_t<float>>();
        auto scores = out["scores"].cast<py::array_t<float>>();
        auto class_id = out["class_id"].cast<py::array_t<int>>();
        auto kps_obj = out["keypoints"];
        auto masks_obj = out["masks"];
        auto obb_obj = out["obb"];

        /* ----------- Рисование -------------------------------------------------- */
        if (task == "detect" || task == "pose" || task == "segment") {
            auto b = boxes.unchecked<2>();
            for (ssize_t i = 0; i < boxes.shape(0); ++i)
                cv::rectangle(color_image_,
                              { int(b(i,0)), int(b(i,1)) },
                              { int(b(i,2)), int(b(i,3)) },
                              { 0, 255, 0 }, 2);
        }

        if (task == "pose" && py::isinstance<py::array>(kps_obj)) {
            auto kps = kps_obj.cast<py::array_t<float>>();
            auto kp  = kps.unchecked<3>();
            for (ssize_t n = 0; n < kps.shape(0); ++n) {
                for (int i = 0; i < 17; ++i) {
                    int x=int(kp(n,i,0)), y=int(kp(n,i,1));
                    if (x==0 && y==0) continue;
                    cv::circle(color_image_, {x,y}, 3, {0,255,255}, -1);
                }
                for (auto [a,b] : COCO_PAIRS_) {
                    if ((kp(n,a,0)||kp(n,a,1)) && (kp(n,b,0)||kp(n,b,1)))
                        cv::line(color_image_,
                                {int(kp(n,a,0)),int(kp(n,a,1))},
                                {int(kp(n,b,0)),int(kp(n,b,1))},
                                {0,255,0},2);
                }
            }
        }

        if (task == "segment" && py::isinstance<py::array>(masks_obj)) {
            auto masks = masks_obj.cast<py::array_t<bool>>();     // (N,H,W)
            auto m     = masks.unchecked<3>();
            for (ssize_t n=0;n<masks.shape(0);++n) {
                cv::Mat mask(rgb.rows,rgb.cols,CV_8U,(void*)m.data(n,0,0));
                cv::Mat colored; cv::cvtColor(mask*255,colored,cv::COLOR_GRAY2BGR);
                cv::addWeighted(color_image_,1.0,colored,0.5,0,color_image_);
            }
        }

        if (task == "obb" && py::isinstance<py::array>(obb_obj)) {
            auto obb = obb_obj.cast<py::array_t<float>>();        // (N,5) cx,cy,w,h,theta
            auto o   = obb.unchecked<2>();
            for (ssize_t i=0;i<obb.shape(0);++i){
                cv::RotatedRect rr({o(i,0),o(i,1)},{o(i,2),o(i,3)}, o(i,4)*180/M_PI);
                cv::Point2f pts[4]; rr.points(pts);
                for(int j=0;j<4;++j)
                    cv::line(color_image_,pts[j],pts[(j+1)%4],{0,255,0},2);
            }
        }

        iiwa_realsense_interfaces::msg::Detections detection_msg_;
        detection_msg_.header.stamp = ros_time;
        detection_msg_.header.frame_id = camera_name_;
        detection_msg_.task = task;

        const size_t N = boxes.shape(0);
        detection_msg_.num_detections  = static_cast<uint32_t>(N);

        detection_msg_.boxes_xyxy.assign(boxes.data(), boxes.data()+boxes.size());
        detection_msg_.scores.assign(scores.data(), scores.data()+scores.size());
        detection_msg_.class_id.assign(class_id.data(), class_id.data()+class_id.size());

        if (py::isinstance<py::array>(kps_obj)) {
            auto kps = kps_obj.cast<py::array_t<float>>();
            detection_msg_.keypoints_xy.assign(kps.data(), kps.data()+kps.size());
        } else {
            detection_msg_.keypoints_xy.clear();
        }

        if (py::isinstance<py::array>(masks_obj)) {
            auto buf = masks_obj.cast<py::array>();
            py::buffer_info info = buf.request();
            detection_msg_.masks.resize(info.size);
            std::memcpy(detection_msg_.masks.data(), info.ptr, info.size);
        } else {
            detection_msg_.masks.clear();
        }

        if (py::isinstance<py::array>(obb_obj)) {
            auto obb = obb_obj.cast<py::array_t<float>>();
            detection_msg_.obb_xywhr.assign(obb.data(), obb.data()+obb.size());
        } else {
            detection_msg_.obb_xywhr.clear();
        }

        detection_publisher_->publish(detection_msg_);
      }

      if (publish_image_) {
        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = ros_time;
        msg.header.frame_id = camera_name_;
        msg.format = "jpeg";

        std::vector<uchar> buf;
        cv::imencode(".jpg", color_image_, buf);
        msg.data = buf;
        publisher_->publish(msg);
      }
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
        
        string msg = "Neural network status: ";
        msg += publish_detect_object_ ? "on" : "off";
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
        response->status  = false;
        response->message = "device must be 'cpu' or 'cuda'";
        RCLCPP_WARN(get_logger(), response->message.c_str());
        return;
      }

      if (request->max_objects == 0) {
        response->status  = false;
        response->message = "max_objects must be > 0";
        RCLCPP_WARN(get_logger(), response->message.c_str());
        return;
      }

      model_conf_ = request->confidence;
      model_device_ = request->device;
      max_object_detection_ = request->max_objects;

      bool publish_status = publish_detect_object_; 
      publish_detect_object_ = false;
      
      py::dict out = load_model_func_(request->task, request->model_name, request->model_format).cast<py::dict>();
      bool status = out["status"].cast<py::bool_>();
      string message = out["message"].cast<py::str>();

      RCLCPP_INFO(this->get_logger(), message.c_str());
      
      response->status = status;
      response->message = message;

      publish_detect_object_ = publish_status;
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
      if (status) RCLCPP_INFO(this->get_logger(), "Profile changed");
      else RCLCPP_WARN(this->get_logger(), "Failed to change profile");
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

      string msg = "Crop updated";
      RCLCPP_INFO(this->get_logger(), msg.c_str());
      response->success = true;
      response->message = msg;
    }


    // Service callback:
    // - Returns depth value at (x, y) pixel coordinate
    // - Validates bounds
    // - Reads depth from latest frame and returns it
    void handle_get_depth_at_point(const std::shared_ptr<iiwa_realsense_interfaces::srv::GetDepthAtPoint::Request> request,
                                 std::shared_ptr<iiwa_realsense_interfaces::srv::GetDepthAtPoint::Response> response) {
      try {
        uint16_t x = request->x;
        uint16_t y = request->y;
        string msg = "Depth successfully obtained";

        RCLCPP_INFO(this->get_logger(), "Image size: cols=%d, rows=%d; Request: x=%d, y=%d",
                    depth_image_.cols, depth_image_.rows, request->x, request->y);


        if (x >= depth_image_.cols || y >= depth_image_.rows) {
          msg = "Coordinates out of bounds";
          RCLCPP_WARN(this->get_logger(), msg.c_str());
          response->success = false;
          response->message = msg;
          response->depth = 0.0;
          return;
        }

        RCLCPP_INFO(this->get_logger(), msg.c_str());

        uint16_t raw_depth = depth_image_.at<uint16_t>(y, x);
        response->depth = static_cast<float>(raw_depth);
        response->success = true;
        response->message = msg;

      } catch (const exception& e) {
        RCLCPP_ERROR(this->get_logger(), e.what());
        response->success = false;
        response->message = e.what();
        response->depth = 0.0;
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


    // --- Internal parameters and member variables ---
    float zoom_level_;
    float model_conf_;
    string camera_name_;
    string model_device_;
    py::object detect_func_;
    py::object load_model_func_;
    uint8_t max_object_detection_;
    uint16_t width_, height_, fps_;
    string model_name, model_format;
    cv::Mat depth_image_, color_image_;
    uint16_t crop_x_offset_, crop_y_offset_;
    bool publish_image_, publish_detect_object_;

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
    rclcpp::Publisher<iiwa_realsense_interfaces::msg::Detections>::SharedPtr detection_publisher_;

    // ROS services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr nn_detection_service_;
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