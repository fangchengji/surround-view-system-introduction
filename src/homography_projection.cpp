
#include <algorithm>
#include "homography_projection.h"

namespace svs {

void generate_grid(cv::Mat& grid, float x_start, float x_stop, float x_step, float y_start, float y_stop, float y_step) {
  std::vector<float> xs = arange(x_start, x_stop, x_step);
  std::vector<float> ys = arange(y_start, y_stop, y_step);
  cv::Mat xs_mat = cv::Mat(ys.size(), xs.size(), CV_32FC1);
  for (int i = 0; i < ys.size(); ++i) {
    std::copy(xs.begin(), xs.end(), xs_mat.ptr<float>() + i * xs.size());
  }

  cv::Mat ys_mat = cv::Mat(xs.size(), ys.size(), CV_32FC1);
  for (int i = 0; i < xs.size(); ++i) {
    std::copy(ys.begin(), ys.end(), ys_mat.ptr<float>() + i * ys.size());
  }

  cv::Mat ones = cv::Mat::ones(ys.size(), xs.size(), CV_32FC1);
  std::vector<cv::Mat> channels;
  channels.push_back(xs_mat);
  channels.push_back(ys_mat.t());
  channels.push_back(ones);
  
  cv::merge(channels, grid);
}

void generate_grid(cv::Mat& grid, const std::vector<float>& xs, const std::vector<float>& ys) {
  cv::Mat xs_mat = cv::Mat(ys.size(), xs.size(), CV_32FC1);
  for (int i = 0; i < ys.size(); ++i) {
    std::copy(xs.begin(), xs.end(), xs_mat.ptr<float>() + i * xs.size());
  }

  cv::Mat ys_mat = cv::Mat(xs.size(), ys.size(), CV_32FC1);
  for (int i = 0; i < xs.size(); ++i) {
    std::copy(ys.begin(), ys.end(), ys_mat.ptr<float>() + i * ys.size());
  }

  cv::Mat ones = cv::Mat::ones(ys.size(), xs.size(), CV_32FC1);
  std::vector<cv::Mat> channels;
  channels.push_back(xs_mat);
  channels.push_back(ys_mat.t());
  channels.push_back(ones);
  
  cv::merge(channels, grid);
}

void homography_projection_lut(const cv::Mat& src_lut, cv::Mat& dst_lut, const cv::Mat& H,  
	const std::vector<float>& xs, const std::vector<float>& ys) {
  dst_lut = cv::Mat::zeros(cv::Size(xs.size(), ys.size()), CV_32FC2);
  for (int j = 0; j < ys.size(); ++j) {
    for (int i = 0; i < xs.size(); ++i) {
      cv::Mat src = H * (cv::Mat_<double>(3, 1) << xs[i], ys[j], 1.0f);
      src.at<double>(0, 0) = src.at<double>(0, 0) / src.at<double>(2, 0);
      src.at<double>(1, 0) = src.at<double>(1, 0) / src.at<double>(2, 0);
      if (src.at<double>(0, 0) >= 0 && src.at<double>(0, 0) < src_lut.cols 
        && src.at<double>(1, 0) >= 0 && src.at<double>(1, 0) < src_lut.rows) {
        dst_lut.at<cv::Vec2f>(j, i) = bilinear_interpolation<cv::Vec2f>(cv::Point2f(src.at<double>(0, 0), src.at<double>(1, 0)), src_lut);
      }
    }
  }
}

void homography_projection_lut_cv(const cv::Mat& src_lut, cv::Mat& dst_lut, const cv::Mat& H,  
	const std::vector<float>& xs, const std::vector<float>& ys) {
  // use opencv API, 40 times faster than hard coding
  cv::Mat grid;
  generate_grid(grid, xs, ys);
  cv::Mat dst;
  cv::transform(grid, dst, H);
  std::vector<cv::Mat> channels;
  cv::split(dst, channels);
  channels[0] /= channels[2];
  channels[1] /= channels[2];
  cv::remap(src_lut, dst_lut, channels[0], channels[1], cv::INTER_LINEAR);
}

}	// namespace svs