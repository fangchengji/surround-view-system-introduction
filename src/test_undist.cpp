#include <iostream>

#include <opencv2/opencv.hpp>

#include "distortion_provider.hpp"
#include "distortion_correction.h"


int main( int argc, char** argv ) {
  if( argc != 2) {
      std::cout <<" Usage: display_image ImageToLoadAndDisplay" << std::endl;
      return -1;
  }

  cv::Mat img;
  img = cv::imread(argv[1], cv::IMREAD_COLOR);   // Read the file

  // test undistortion
  auto dp_ptr = std::make_shared<svs::DistortionProvider>("../x01_avm/x01_distortion.txt");
  // cv::Mat undist_img;
  // svs::undistort_image(img, undist_img, cv::Size(1350, 720), 
  //                      0.0042, 0.00975, cv::Point2f(1280/2.0, 720/2.0), dp_ptr);
  // cv::imwrite("undist.jpg", undist_img);

  // lut distortion
  // cv::Mat lut;
  // svs::calculate_undistortion_lut(cv::Size(1920, 1280), cv::Size(3770, 2030), 
  //                      0.003, 0.004, cv::Point2f(1920/2.0, 1280/2.0), dp_ptr, lut);
  cv::Mat map1, map2;
  cv::Mat k = (cv::Mat_<float>(3, 3) << 
    423.333333333, 0.0, 960,
    0.0, 423.333333333, 640,
    0, 0, 1 );
  cv::Mat p = ( cv::Mat_<float>(3, 3) <<
    3.169e+02, 0., 1.8861473972788158e+03, 
    0.,3.169e+02, 1.0155844789203933e+03, 
    0., 0., 1. );
  cv::Size undist_size(3770, 2030);
  cv::Mat undist_img;

  svs::calculate_undistortion_lut(k, dp_ptr, p, undist_size, map1, map2);
  svs::undistort_image(img, undist_img, map1, map2);
  cv::imwrite("undist.jpg", undist_img);

  return 0;
}


