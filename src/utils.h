#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace svs {

template<typename T>
inline std::vector<T> arange(T start, T stop, T step) {
  std::vector<T> values;
  int len = int((stop - start) / step);
  double half_step = step / 2.0;
  for (int i = 0; i < len; ++i) {
    values.push_back(start + i * step + half_step);
  }
  return values;
}

template <typename T>
inline T bilinear_interpolation(const cv::Point2f& pt, const cv::Mat& src) {
  int x0 = std::max(0, std::min(src.cols - 1, (int)floor(pt.x)));
  int y0 = std::max(0, std::min(src.rows - 1, (int)floor(pt.y)));
  int x1 = std::max(0, std::min(src.cols - 1, x0 + 1));
  int y1 = std::max(0, std::min(src.rows - 1, y0 + 1));

  const T& pt0(src.at<T>(y0, x0));
  const T& pt1(src.at<T>(y0, x1));
  const T& pt2(src.at<T>(y1, x0));
  const T& pt3(src.at<T>(y1, x1));

  T v0 = (x1 - pt.x) * pt0 + (pt.x - x0) * pt1;
  T v1 = (x1 - pt.x) * pt2 + (pt.x - x0) * pt3;
  return (y1 - pt.y) * v0 + (pt.y - y0) * v1;
}

}	// namespace svs