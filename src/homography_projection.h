#pragma once 

#include "utils.h"

#include <opencv2/opencv.hpp>

namespace svs {

void homography_projection_lut(const cv::Mat& src_lut, cv::Mat& dst_lut, const cv::Mat& H,  
	const std::vector<float>& xs, const std::vector<float>& ys);

void homography_projection_lut_cv(const cv::Mat& src_lut, cv::Mat& dst_lut, const cv::Mat& H,  
	const std::vector<float>& xs, const std::vector<float>& ys);

void generate_grid(cv::Mat& grid, float x_start, float x_stop, float x_step, float y_start, float y_stop, float y_step);

void generate_grid(cv::Mat& grid, const std::vector<float>& xs, const std::vector<float>& ys);

}	// namespace svs