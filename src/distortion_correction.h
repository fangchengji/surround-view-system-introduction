#pragma once
#include "distortion_provider.hpp"
#include "utils.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace svs {

void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size, float origin_pixel_size, 
  float new_pixel_size, const cv::Point2f& center, const std::shared_ptr<DistortionProvider> dp);
  
void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Mat& lut);
void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map1, cv::Mat& map2);

void calculate_undistortion_lut(const cv::Size& in_size, const cv::Size& out_size, float new_pixel_size, 
  float origin_pixel_size, const cv::Point2f& center, std::shared_ptr<DistortionProvider> dp, cv::Mat& res);
void calculate_undistortion_lut(cv::Mat K, std::shared_ptr<DistortionProvider> dp, cv::Mat P,
                                             const cv::Size &size, cv::Mat &map1, cv::Mat &map2);
}	 // namespace svs