#pragma once

#include "homography_projection.h"
#include "distortion_correction.h"
#include <chrono>

#define IMGW 1280
#define IMGH 720
#define PIXEL_SIZE 0.0042     // 4.2 um

#define CORRECTW 1350
#define CORRECTH 720
#define CORRECTPS 0.00975     // 9.75 um

#define SVS_CAMERA_NUM 4

#define SIDE_DISTANCE 8000
#define FRONT_DISTANCE 6000
#define REAR_DISTANCE 8000
#define W_OFFSET 50

#define VEHICLE_L 5023
#define VEHICLE_W 1960
#define HALF_VEHICLE_L VEHICLE_L / 2
#define HALF_VEHICLE_W VEHICLE_W / 2
#define AVM_PIXEL_SIZE 15 

#define DISTORTION_FILE "../data/m01_distortion.txt"

namespace svs {

template<typename T>
struct Interval {
  T start;
  T stop;
  T step;
  Interval(T _start, T _stop, T _step) : start(_start), stop(_stop), step(_step) {}; 
  std::vector<T> elements() {return arange<T>(start, stop, step); }
};

class SurroundViewStitcher {
public:
  SurroundViewStitcher() {
    // TODO: set all params by config file.

    camera_pixel_size_ = PIXEL_SIZE;
    correct_pixel_size_ = CORRECTPS;
    image_size_ = cv::Size(IMGW, IMGH);
    correct_size_ = cv::Size(CORRECTW, CORRECTH);

    for (int i = 0; i < SVS_CAMERA_NUM; ++i) {
      img_centers_.push_back(cv::Point2f(IMGW/2.0, IMGH/2.0));
    }

    dp_ = std::make_shared<DistortionProvider>(DISTORTION_FILE);
    if (dp_ == nullptr) {
      printf("DistortionProvider make failed!!!\n");
    }

    // front
    x_intervals_.emplace_back(Interval<float>(-(SIDE_DISTANCE + HALF_VEHICLE_W) + W_OFFSET, 
                                    SIDE_DISTANCE + HALF_VEHICLE_W + W_OFFSET, AVM_PIXEL_SIZE));
    y_intervals_.emplace_back(Interval<float>(FRONT_DISTANCE + HALF_VEHICLE_L, 
                                    HALF_VEHICLE_L, -AVM_PIXEL_SIZE));
    cv::Mat front_H = (cv::Mat_<double>(3, 3) <<
        -7.7701834498851527e-02, -3.1444351316730390e-01,
        6.7667562256659016e+02, -7.5324831312253700e-04,
        -1.3769149770759789e-01, 2.4015947063634835e+02,
        -9.9737542400689613e-06, -4.6841167493419367e-04, 1.);
    Hs_.push_back(front_H);
    
    // rear
    x_intervals_.emplace_back(Interval<float>(-(SIDE_DISTANCE + HALF_VEHICLE_W) + W_OFFSET, 
      SIDE_DISTANCE + HALF_VEHICLE_W + W_OFFSET, AVM_PIXEL_SIZE));
    y_intervals_.emplace_back(Interval<float>(-HALF_VEHICLE_L, -(HALF_VEHICLE_L + REAR_DISTANCE), -AVM_PIXEL_SIZE));
    cv::Mat rear_H = (cv::Mat_<double>(3, 3) << 
        9.6078919872715829e-02, 3.7984478412119704e-01, 6.7176811959932411e+02,
        -1.0273559647656078e-04, 1.5443579517981729e-01,
        1.6100076711089713e+02, 7.2171039227399751e-06, 5.6609934109434605e-04,
        1.);
    Hs_.push_back(rear_H);

    // left
    x_intervals_.emplace_back(Interval<float>(-(SIDE_DISTANCE + HALF_VEHICLE_W) + W_OFFSET, 
      -HALF_VEHICLE_W + W_OFFSET, AVM_PIXEL_SIZE));
    y_intervals_.emplace_back(Interval<float>(HALF_VEHICLE_L + FRONT_DISTANCE, 
      -(HALF_VEHICLE_L + REAR_DISTANCE), -AVM_PIXEL_SIZE));
    cv::Mat left_H = (cv::Mat_<double>(3, 3) << -6.4452076979484758e+01, 2.0309867965189003e+01,
        -1.1337185738536151e+04, -2.3134388302146341e+01,
        6.8264561066865570e-01, 2.8003769307996728e+04,
        -9.6754821267343902e-02, 3.9264130206281816e-03, 1.);
    Hs_.push_back(left_H);

    // right
    x_intervals_.emplace_back(Interval<float>(HALF_VEHICLE_W + W_OFFSET, 
      SIDE_DISTANCE + HALF_VEHICLE_W + W_OFFSET, AVM_PIXEL_SIZE));
    y_intervals_.emplace_back(Interval<float>(HALF_VEHICLE_L + FRONT_DISTANCE, 
      -(HALF_VEHICLE_L + REAR_DISTANCE), -AVM_PIXEL_SIZE));
    cv::Mat right_H = (cv::Mat_<double>(3, 3) << -2.3861579738693353e+00, 5.9431137723140892e-01,
        2.6688519339014096e+02, -9.3666348357416507e-01,
        -7.1126631549249227e-03, -5.6139650014375593e+02,
        -3.5400364043628061e-03, 1.3720983106096465e-05, 1.);
    Hs_.push_back(right_H);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    generate_svs_luts(geo_lut_, idx_lut_);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    printf("generate_svs_luts time: %ld\n", duration.count());
  }

  virtual ~SurroundViewStitcher() = default;

  void generate_undist_luts(std::vector<cv::Mat>& dst_luts) {
    dst_luts.clear();
    for (int i = 0; i < SVS_CAMERA_NUM; ++i) {
      cv::Mat undist_lut;
      calculate_undistortion_lut(image_size_, correct_size_, camera_pixel_size_, correct_pixel_size_, 
        img_centers_[i], dp_, undist_lut);
      dst_luts.push_back(undist_lut);
    }
  }

  void generate_project_luts(const std::vector<cv::Mat>& src_luts, std::vector<cv::Mat>& dst_luts) {
    dst_luts.clear();
    for (int i = 0; i < SVS_CAMERA_NUM; ++i) {
      cv::Mat project_lut;
      std::vector<float> xs = x_intervals_[i].elements();
      std::vector<float> ys = y_intervals_[i].elements();
      homography_projection_lut_cv(src_luts[i], project_lut, Hs_[i], xs, ys);
      dst_luts.push_back(project_lut);
    }
  }

  void generate_stitch_luts(const std::vector<cv::Mat>& src_luts, cv::Mat& dst_lut, cv::Mat& idx_lut) {
    const cv::Mat& front_lut = src_luts[0];
    const cv::Mat& rear_lut = src_luts[1];
    const cv::Mat& left_lut = src_luts[2];
    const cv::Mat& right_lut = src_luts[3];
    assert(front_lut.cols == rear_lut.cols && left_lut.rows == right_lut.rows);

    int shift_w = front_lut.cols - right_lut.cols;
    int shift_h = left_lut.rows - rear_lut.rows;
    int width_out = front_lut.cols;
    int height_out = left_lut.rows;
    dst_lut = cv::Mat::zeros(cv::Size(width_out, height_out), CV_32FC2);
    idx_lut = cv::Mat::zeros(cv::Size(width_out, height_out), CV_8UC1);

    float angle_ratio = 3.0;
    for (int j = 0; j < height_out; ++j) {
      for (int i = 0; i < width_out; ++i) {
        // top part
        if (j < front_lut.rows) {
          // about 18 degree
          if (i < left_lut.cols && (front_lut.rows - 1 -j) <= (left_lut.cols - 1 -i) * angle_ratio) {
            dst_lut.at<cv::Vec2f>(j, i) = left_lut.at<cv::Vec2f>(j, i);
            idx_lut.at<uint8_t>(j, i) = 2;
          } else if (i >= shift_w && (front_lut.rows -1 -j) <= (i - shift_w) * angle_ratio) {
            dst_lut.at<cv::Vec2f>(j, i) = right_lut.at<cv::Vec2f>(j, i - shift_w);
            idx_lut.at<uint8_t>(j, i) = 3;
          } else {
            dst_lut.at<cv::Vec2f>(j, i) = front_lut.at<cv::Vec2f>(j, i);
            idx_lut.at<uint8_t>(j, i) = 0;
          }
        // middle part
        } else if (j >= front_lut.rows && j < shift_h) {
          if (i < left_lut.cols) {
            dst_lut.at<cv::Vec2f>(j, i) = left_lut.at<cv::Vec2f>(j, i);
            idx_lut.at<uint8_t>(j, i) = 2;
          } else if (i >= shift_w) {
            dst_lut.at<cv::Vec2f>(j, i) = right_lut.at<cv::Vec2f>(j, i - shift_w);
            idx_lut.at<uint8_t>(j, i) = 3;
          }
        // bottom part
        } else {
          if (i < left_lut.cols && (left_lut.cols - 1 - i) > (j - shift_h) * angle_ratio) {
            dst_lut.at<cv::Vec2f>(j, i) = left_lut.at<cv::Vec2f>(j, i);
            idx_lut.at<uint8_t>(j, i) = 2;
          } else if (i >= shift_w && (i - shift_w) > (j - shift_h) * angle_ratio) {
            dst_lut.at<cv::Vec2f>(j, i) = right_lut.at<cv::Vec2f>(j, i - shift_w);
            idx_lut.at<uint8_t>(j, i) = 3;
          } else {
            dst_lut.at<cv::Vec2f>(j, i) = rear_lut.at<cv::Vec2f>(j - shift_h, i);
            idx_lut.at<uint8_t>(j, i) = 1;
          }
        }
      }
    }
  }

  void generate_svs_luts( cv::Mat& dst_lut, cv::Mat& idx_lut) {
    std::vector<cv::Mat> undist_luts;
    auto t0 = std::chrono::high_resolution_clock::now();
    generate_undist_luts(undist_luts);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    printf("generate_undist_luts time: %ld\n", duration.count());

    t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> project_luts;
    generate_project_luts(undist_luts, project_luts);
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    printf("generate_project_luts time: %ld\n", duration.count());

    t0 = std::chrono::high_resolution_clock::now();
    generate_stitch_luts(project_luts, dst_lut, idx_lut);
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    printf("generate_stitch_luts time: %ld\n", duration.count());
  }

  void stitch(const std::vector<cv::Mat>& imgs, cv::Mat& dst, const cv::Mat& geo_lut, const cv::Mat& idx_lut) {
    assert(imgs.size() == 4);
    assert(geo_lut.rows == idx_lut.rows && geo_lut.cols == idx_lut.cols);

    dst = cv::Mat::zeros(cv::Size(geo_lut.cols, geo_lut.rows), CV_8UC3);
    for (int j = 0; j < geo_lut.rows; ++j) {
      for (int i = 0; i < geo_lut.cols; ++i) {
        cv::Vec2f pt = geo_lut.at<cv::Vec2f>(j, i);
        if (pt[0] >= 0 && pt[0] < imgs[0].cols && pt[1] >= 0 && pt[1] < imgs[0].rows) {
          dst.at<cv::Vec3b>(j, i) = imgs[idx_lut.at<uint8_t>(j, i)].at<cv::Vec3b>(int(pt[1]), int(pt[0]));
        }
      }
    }
  }

  void run(const std::vector<cv::Mat>& imgs, cv::Mat& dst) {
    assert(imgs.size() == 4);
    auto t0 = std::chrono::high_resolution_clock::now();
    stitch(imgs, dst, geo_lut_, idx_lut_);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    printf("stitch time: %ld\n", duration.count());
  }

  void get_lut(cv::Mat& geo_lut, cv::Mat& idx_lut) {
    geo_lut = geo_lut_.clone();
    idx_lut = idx_lut_.clone();
  }

private:
  float camera_pixel_size_;
  float correct_pixel_size_;
  cv::Size image_size_;
  cv::Size correct_size_;

  std::vector<cv::Point2f> img_centers_;
  std::vector<cv::Mat> Hs_;
  std::vector<Interval<float>> x_intervals_;
  std::vector<Interval<float>> y_intervals_;

  cv::Mat geo_lut_;
  cv::Mat idx_lut_;

  std::shared_ptr<DistortionProvider> dp_;
};

} // namespace svs