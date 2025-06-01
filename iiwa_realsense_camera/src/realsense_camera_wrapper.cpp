#include "iiwa_realsense_camera/realsense_camera_wrapper.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = filesystem;


// Constructor: Initializes RealSense pipeline, checks for RGB sensor, and sets stream profiles.
// Throws runtime_error if RGB camera not found.
RealsenseCameraWrapper::RealsenseCameraWrapper(uint16_t& width, uint16_t& height, uint16_t& fps)
    : align_(RS2_STREAM_COLOR), pipeline_started_(false) {
        rs2::pipeline_profile pipeline_profile = pipeline_.start(config_);
        pipeline_.stop();
        device_ = pipeline_profile.get_device();

        // Set max distance on supported sensors (optional).
        for (auto&& sensor : device_.query_sensors()) {
            if (sensor.supports(RS2_OPTION_MAX_DISTANCE)) {
                sensor.set_option(RS2_OPTION_MAX_DISTANCE, 10.0f);
            }
        }

        // Ensure the device has an RGB camera.
        bool found_rgb = false;
        for (auto&& s : device_.query_sensors()) {
            if (s.get_info(RS2_CAMERA_INFO_NAME) == string("RGB Camera")) {
                found_rgb = true;
                break;
            }
        }

        if (!found_rgb) {
            string message = "[ERROR] Depth camera with Color sensor is required.";
            log_info(message);
            throw runtime_error(message);
        }

        // Enable color and depth streams with given resolution and fps.
        config_.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
        config_.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
    }

// Destructor: Stops streaming and resets hardware (soft reboot).
RealsenseCameraWrapper::~RealsenseCameraWrapper() {
    stopStreaming();
    try {
        device_.hardware_reset();
        log_info("[INFO] Camera rebooted");
    } catch (...) {}
}

// Starts the camera stream. If already running, does nothing.
string RealsenseCameraWrapper::startStreaming() {
    if (pipeline_started_) return "Camera already started";
    try
    {
        pipeline_.start(config_);
        pipeline_started_ = true;
    }
    catch(const exception& e) {
        log_message(e.what());
    };

    log_info("[INFO] Camera started");
    return "Camera started";
}

// Stops the camera stream. If already stopped, does nothing.
string RealsenseCameraWrapper::stopStreaming() {
    if (!pipeline_started_) return "Camera already stopped";
    try {
        pipeline_.stop();
        pipeline_started_ = false;
    } catch (const exception& e) {
        log_message(e.what());
    }
    
    log_info("[INFO] Camera stopped");
    return "Camera stopped";
}

// Tries to change the camera's resolution and fps.
// Only changes if both RGB and depth sensors support requested profile.
bool RealsenseCameraWrapper::changeCameraProfile(uint16_t& width, uint16_t& height, uint16_t& fps) {
    auto depth_profiles = getProfiles("Stereo Module");
    auto color_profiles = getProfiles("RGB Camera");

    if (checkProfileValidity(depth_profiles, width, height, fps) &&
        checkProfileValidity(color_profiles, width, height, fps)) {
            stopStreaming();
            config_ = rs2::config();
            config_.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
            config_.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
            startStreaming();

            log_info("[INFO] Camera profile changed successfully. New settings: Width=", width,
                ", Height=", height, ", FPS=", fps);
            return true;
        }
    
    log_info("[WARN] Failed to change camera profile. Requested settings: Width=", width,
        ", Height=", height, ", FPS=", fps);
    return false;
}

// Captures one frame and returns aligned depth and color images as OpenCV matrices.
pair<cv::Mat, cv::Mat> RealsenseCameraWrapper::getAlignedImages() {
    if (!pipeline_started_) startStreaming();
    rs2::frameset frames = pipeline_.wait_for_frames();
    frames = align_.process(frames);

    rs2::depth_frame depth_frame = frames.get_depth_frame();
    rs2::video_frame color_frame = frames.get_color_frame();

    if (!depth_frame || !color_frame) {
        string message = "[ERROR] Invalid frames received";
        log_info(message);
        throw std::runtime_error(message);
    }

    cv::Mat depth_image(cv::Size(depth_frame.get_width(), depth_frame.get_height()),
                        CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()),
                        CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

    return {depth_image, color_image};
}

// Gets the most recent aligned frameset.
rs2::frameset RealsenseCameraWrapper::getLatestFrameset() {
    if (!pipeline_started_) startStreaming();
    rs2::frameset frames = pipeline_.wait_for_frames();
    return align_.process(frames);
}

// Builds a 3D point cloud and corresponding texture coordinates from given frames.
pair<vector<array<float, 3>>, vector<array<float, 2>>> RealsenseCameraWrapper::getPointCloud(
    const rs2::depth_frame& depth_frame, const rs2::video_frame color_frame) {
    
    rs2::points points = rs_pointcloud_.calculate(depth_frame);
    rs_pointcloud_.map_to(color_frame);

    const rs2::vertex* vertices = points.get_vertices();
    const rs2::texture_coordinate* tex_coords = points.get_texture_coordinates();

    size_t size = points.size();
    vector<array<float, 3>> verts(size);
    vector<array<float, 2>> tex(size);

    for (size_t i = 0; i < size; ++i) {
        verts[i] = {vertices[i].x, vertices[i].y, vertices[i].z};
        tex[i] = {tex_coords[i].u, tex_coords[i].v};
    }

    return {verts, tex};
}

// Collects and returns camera info: name, firmware, USB type, and supported profiles.
CameraInfo RealsenseCameraWrapper::getCameraInformation() {
    return CameraInfo{
        device_.get_info(RS2_CAMERA_INFO_NAME),
        device_.get_info(RS2_CAMERA_INFO_PRODUCT_LINE),
        device_.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER),
        device_.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION),
        device_.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR),
        getProfiles("Stereo Module"),
        getProfiles("RGB Camera")
    };
}

// Returns all profile strings (as JSON) for a sensor by name.
vector<string> RealsenseCameraWrapper::getProfiles(const string& sensor_name_filter) {
    vector<string> profiles;
    for (auto&& sensor : device_.query_sensors()) {
        if (sensor.get_info(RS2_CAMERA_INFO_NAME) == sensor_name_filter) {
            auto stream_profiles = getSensorProfiles(sensor);
            profiles.insert(profiles.end(), stream_profiles.begin(), stream_profiles.end());
        }
    }
    return profiles;
}

// Converts RealSense stream profiles of a sensor into JSON strings with dimensions, FPS, and format.
vector<string> RealsenseCameraWrapper::getSensorProfiles(rs2::sensor& sensor) {
    vector<string> stream_profiles;
    auto profiles = sensor.get_stream_profiles();

    for (auto&& profile : profiles) {
        if (auto vsp = profile.as<rs2::video_stream_profile>()) {
            json j;
            rs2_stream stream_type = vsp.stream_type();
            rs2_format format = vsp.format();

            j["stream_type"] = rs2_stream_to_string(stream_type);

            j["width"] = vsp.width();
            j["height"] = vsp.height();
            j["fps"] = vsp.fps();
            
            j["format"] = rs2_format_to_string(format);
            stream_profiles.push_back(j.dump());
        }
    }

    return stream_profiles;
}

// Validates if given resolution/fps exists in the sensor's profile list.
bool RealsenseCameraWrapper::checkProfileValidity(const vector<string>& profiles, uint16_t& width, uint16_t& height, uint16_t& fps) {
    for (const auto& profile_str : profiles) {
        auto j = json::parse(profile_str);
        if (j["width"] == width && j["height"] == height && j["fps"] == fps) {
            return true;
        }
    }
    return false;
}

// Logs a message to log/RealsenseCameraWrapper.log with timestamp.
void RealsenseCameraWrapper::log_message(const string& message) {
    const string log_dir = "log";
    const string log_file = log_dir + "/RealsenseCameraWrapper.log";

    if (!fs::exists(log_dir)) {
        fs::create_directory(log_dir);
    }

    std::ofstream ofs(log_file, std::ios::app);
    if (ofs.is_open()) {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        ofs << "[" << std::put_time(std::localtime(&now), "%F %T") << "] " << message << std::endl;
    }
}

// Accepts multiple arguments, concatenates into a single string, and logs it.
template<typename... Args>
void RealsenseCameraWrapper::log_info(Args... args) {
    std::ostringstream oss;
    (oss << ... << args);
    log_message(oss.str());
}