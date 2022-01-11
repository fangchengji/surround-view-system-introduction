#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>

#include "surround_view_stitcher_x01.hpp"
#include "stitching.h"

int main( int argc, char** argv ) {
  std::string base_dir = "../x01_avm/";
  if( argc != 2) {
    std::cout << "Usage: test_stitch <img_base_dir>" << std::endl;
    std::cout << "Use default base dir: ../x01_avm/" << std::endl;
  } else {
    base_dir = argv[1];
  }

  std::vector<cv::Mat> imgs;
  cv::Mat img;
  std::vector<std::string> img_names = {"front.bmp", "rear.bmp", "left.bmp", "right.bmp"};
  for (int i = 0; i < img_names.size(); ++i) {
    img = cv::imread(base_dir + img_names[i], cv::IMREAD_COLOR);   // Read the file
    imgs.push_back(img);
  }

  // test stitch
  svs::SurroundViewStitcherX01 stitcher;
  cv::Mat out_img;
  // stitcher.run(imgs, out_img);
  cv::Mat geo_lut;
  cv::Mat idx_lut;
  stitcher.get_lut(geo_lut, idx_lut);

  // Allocate device memory.
  unsigned char* d_input = nullptr;
  float* d_geo_lut = nullptr;
  unsigned char* d_idx_lut = nullptr;
  unsigned char* d_output = nullptr;
  const int img_bytes = 3 * imgs[0].rows * imgs[0].cols * sizeof(unsigned char);
  const int input_bytes = imgs.size() * img_bytes;
  const int geo_lut_bytes = 2 * geo_lut.rows * geo_lut.cols * sizeof(float);
  const int idx_lut_bytes = 1 * geo_lut.rows * geo_lut.cols * sizeof(unsigned char);
  const int output_bytes = 3 * geo_lut.rows * geo_lut.cols * sizeof(unsigned char);
  cudaMalloc((void**)&d_input, input_bytes);
  cudaMalloc((void**)&d_geo_lut, geo_lut_bytes);
  cudaMalloc((void**)&d_idx_lut, idx_lut_bytes);
  cudaMalloc((void**)&d_output, output_bytes);

  unsigned char* h_out = new unsigned char[output_bytes/sizeof(unsigned char)];

  // Copy data from input image to device memory.
  for (int i = 0; i < imgs.size(); ++i) {
    cudaMemcpy(d_input + i * img_bytes, imgs[i].ptr<unsigned char>(), img_bytes, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_geo_lut, geo_lut.ptr<unsigned char>(), geo_lut_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx_lut, idx_lut.ptr<unsigned char>(), idx_lut_bytes, cudaMemcpyHostToDevice);

  svs::GpuTimer timer;
  timer.start();
  svs::lut_stitch(d_input, d_output, imgs[0].cols, imgs[0].rows, geo_lut.cols,
                  geo_lut.rows, d_geo_lut, d_idx_lut);
  timer.stop();

  cudaMemcpy(h_out, d_output, output_bytes, cudaMemcpyDeviceToHost);
  std::cout << "gpu process time " << timer.elapsed() << "ms\n";

  cv::Mat final_img(cv::Size(geo_lut.cols,  geo_lut.rows), CV_8UC3, h_out);
  cv::imwrite("stitch.jpg", final_img);

  delete[] h_out;
  // delete[] img_i;
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_geo_lut);
  cudaFree(d_idx_lut);

  return 0;
}


