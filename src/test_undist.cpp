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
  auto dp_ptr = std::make_shared<svs::DistortionProvider>("../data/m01_distortion.txt");
  cv::Mat undist_img;
  svs::undistort_image(img, undist_img, cv::Size(1350, 720), 
                       0.0042, 0.00975, cv::Point2f(1280/2.0, 720/2.0), dp_ptr);
  cv::imwrite("undist.jpg", undist_img);

  return 0;
}


