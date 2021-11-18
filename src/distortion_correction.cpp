
#include "distortion_correction.h"

namespace svs {

void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size, float origin_pixel_size, 
  float new_pixel_size, const cv::Point2f& center, const std::shared_ptr<DistortionProvider> dp) {
  dst = cv::Mat::zeros(out_size, CV_8UC3);
  float pixel_ratio = new_pixel_size / origin_pixel_size;
  float undist_center_x = (out_size.width - 1) / 2.0;
  float undist_center_y = (out_size.height - 1) / 2.0;
  for (int j = 0; j < out_size.height; ++j) {
    for (int i = 0; i < out_size.width; ++i) {
      float x = i - undist_center_x;
      float y = j - undist_center_y;
      float r = new_pixel_size * sqrt(x * x + y * y);
      float dist = 0;
      dp->get(r, dist);
      // float real_r = r * (1 + dist);

      float k = (1 + dist) * pixel_ratio;
      float src_x = x * k + center.x;
      float src_y = y * k + center.y;
      if (src_x >= 0 && src_x < src.cols && src_y >= 0 && src_y < src.rows) {
        dst.at<cv::Vec3b>(j, i) = bilinear_interpolation<cv::Vec3b>(cv::Point2f(src_x, src_y), src);
      } 
    }
  }
}

void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Mat& lut) {
  dst = cv::Mat::zeros(cv::Size(lut.cols, lut.rows), CV_8UC3);
  cv::Mat map2;
  cv::remap(src, dst, lut, map2, cv::INTER_LINEAR);
}

void calculate_undistortion_lut(const cv::Size& in_size, const cv::Size& out_size, float origin_pixel_size, 
  float new_pixel_size, const cv::Point2f& center, std::shared_ptr<DistortionProvider> dp, cv::Mat& res) {
  res = cv::Mat::zeros(out_size, CV_32FC2);
  float pixel_ratio = new_pixel_size / origin_pixel_size;
  float undist_center_x = (out_size.width - 1) / 2.0;
  float undist_center_y = (out_size.height - 1) / 2.0;
  for (int j = 0; j < out_size.height; ++j) {
    for (int i = 0; i < out_size.width; ++i) {
      float x = i - undist_center_x;
      float y = j - undist_center_y;
      float r = new_pixel_size * sqrt(x * x + y * y);
      float dist = 0;
      dp->get(r, dist);
      // float real_r = r * (1 + dist);

      float k = (1 + dist) * pixel_ratio;
      float src_x = x * k + center.x;
      float src_y = y * k + center.y;
      if (src_x >= 0 && src_x < in_size.width && src_y >= 0 && src_y < in_size.height) {
        res.at<cv::Vec2f>(j, i) = cv::Vec2f(src_x, src_y);
      } 
    }
  }
}

// void undistortion_lut_cv(const cv::Size& in_size, const cv::Size& out_size, float origin_pixel_size, 
//   float new_pixel_size, const cv::Point2f& center, std::shared_ptr<DistortionProvider> dp, cv::Mat& res) {
//   res = cv::Mat::zeros(out_size, CV_32FC2);
//   float pixel_ratio = new_pixel_size / origin_pixel_size;
//   float undist_center_x = (out_size.width - 1) / 2.0;
//   float undist_center_y = (out_size.height - 1) / 2.0;
  

//   for (int j = 0; j < out_size.height; ++j) {
//     for (int i = 0; i < out_size.width; ++i) {
//       float x = i - undist_center_x;
//       float y = j - undist_center_y;
//       float r = new_pixel_size * sqrt(x * x + y * y);
//       float dist = 0;
//       dp->get(r, dist);
//       // float real_r = r * (1 + dist);

//       float k = (1 + dist) * pixel_ratio;
//       float src_x = x * k + center.x;
//       float src_y = y * k + center.y;
//       if (src_x >= 0 && src_x < in_size.width && src_y >= 0 && src_y < in_size.height) {
//         res.at<cv::Vec2f>(j, i) = cv::Vec2f(src_x, src_y);
//       } 
//     }
//   }
// }

}	 // namespace svs