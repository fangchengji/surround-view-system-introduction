
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

void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map1, cv::Mat& map2) {
  dst = cv::Mat::zeros(cv::Size(map1.cols, map1.rows), CV_8UC3);
  cv::remap(src, dst, map1, map2, cv::INTER_LINEAR);
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


void calculate_undistortion_lut(cv::Mat K, std::shared_ptr<DistortionProvider> dp, cv::Mat P,
                                             const cv::Size &size, cv::Mat &map1, cv::Mat &map2) {
  // cv::Mat res = cv::Mat::zeros(size, CV_32FC2);
  map1 = cv::Mat::zeros(size, CV_32FC1);
  map2 = cv::Mat::zeros(size, CV_32FC1);
  assert(P.at<float>(0, 0) == P.at<float>(1, 1));         // only for fx == fy
  float x_ratio = K.at<float>(0, 0) / P.at<float>(0, 0);  // K.fx / P.fx
  float y_ratio = K.at<float>(1, 1) / P.at<float>(1, 1);  // K.fy / P.fy
  float undist_center_x = P.at<float>(0, 2);              // cx
  float undist_center_y = P.at<float>(1, 2);              // cy
  for (int j = 0; j < size.height; ++j) {
    for (int i = 0; i < size.width; ++i) {
      float x = i - undist_center_x;
      float y = j - undist_center_y;
      float dist = 0;
      // float r = new_pixel_size * sqrt(x * x + y * y);
      // dp->GetDistort(r, dist);
      float r = sqrt(x * x + y * y);
      float theta = atan2(r, P.at<float>(0, 0));
      dp->get_by_angle(theta, dist);
      // float real_r = r * (1 + dist);

      float src_x = x * (1 + dist) * x_ratio + K.at<float>(0, 2);
      float src_y = y * (1 + dist) * y_ratio + K.at<float>(1, 2);
      // TODO: clip src_x and src_y to source image size
      // res.at<cv::Vec2f>(j, i) = cv::Vec2f(src_x, src_y);
      map1.at<float>(j, i) = src_x;
      map2.at<float>(j, i) = src_y;
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