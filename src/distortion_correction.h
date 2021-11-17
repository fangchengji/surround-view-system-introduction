#pragma once
#include "distortion_provider.hpp"
#include "utils.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace svs {

void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size, float origin_pixel_size, 
  float new_pixel_size, const cv::Point2f& center, const std::shared_ptr<DistortionProvider> dp);

void undistortion_lut(const cv::Size& in_size, const cv::Size& out_size, float new_pixel_size, 
  float origin_pixel_size, const cv::Point2f& center, std::shared_ptr<DistortionProvider> dp, cv::Mat& res);

}	 // namespace svs