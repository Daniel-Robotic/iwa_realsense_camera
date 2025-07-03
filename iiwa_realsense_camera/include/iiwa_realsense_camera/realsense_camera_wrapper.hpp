#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <filesystem>
#include <sstream>

using namespace std;

struct CameraProfile {
    string stream_type;
    int width;
    int height;
    string stream_format;
    int fps;
};

struct CameraInfo {
    string device_name;
    string product_line;
    string serial_number;
    string firmware_version;
    string usb_type;
    vector<std::string> depth_profile;
    vector<std::string> color_profile;
};

class RealsenseCameraWrapper {
    public:
        RealsenseCameraWrapper(uint16_t&, uint16_t&,uint16_t&);
        ~RealsenseCameraWrapper();

        bool changeCameraProfile(uint16_t&, uint16_t&, uint16_t&);
        string startStreaming();
        string stopStreaming();

        tuple<cv::Mat, cv::Mat, rs2_intrinsics> getAlignedImages();
        pair<vector<array<float, 3>>, vector<array<float, 2>>> getPointCloud(const rs2::depth_frame&, const rs2::video_frame);
        rs2::frameset getLatestFrameset();

        CameraInfo getCameraInformation();

    private:
        rs2::pipeline pipeline_;
        rs2::config config_;
        rs2::pointcloud rs_pointcloud_;
        rs2::device device_;
        rs2::align align_;
        bool pipeline_started_;

        vector<string> getProfiles(const string&);
        vector<string> getSensorProfiles(rs2::sensor&);
        bool checkProfileValidity(const vector<string>&, uint16_t&, uint16_t&, uint16_t&);
        void log_message(const string&);
        
        template<typename... Args>
        void log_info(Args... args);
};