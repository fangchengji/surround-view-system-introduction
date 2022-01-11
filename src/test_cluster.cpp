#include <iostream>

#include <opencv2/opencv.hpp>

  // /*计算欧式距离*/
  // float calcuDistance(uchar* ptr, uchar* ptrCen, int cols) {
  //   float d = 0.0;
  //   for (size_t j = 0; j < cols; j++)
  //   {
  //     d += (double)(ptr[j] - ptrCen[j])*(ptr[j] - ptrCen[j]);
  //   }
  //   d = sqrt(d);
  //   return d;
  // }
 
  /*** 
  @brief   最大最小距离算法
  @param data  输入样本数据，每一行为一个样本，每个样本可以存在多个特征数据
  @param Theta 阈值，一般设置为0.5，阈值越小聚类中心越多
  @param centerIndex 聚类中心的下标
  @return 返回每个样本的类别，类别从1开始，0表示未分类或者分类失败
  ***/
  cv::Mat  max_min_dist_clustering(std::vector<cv::Point2d>& pts, float threshold, std::vector<int> center_index) {
    double maxDistance = 0;
    int start = 0;    //初始选一个中心点
    int index = start; //相当于指针指示新中心点的位置
    int k = 0;        //中心点计数，也即是类别
    int num = pts.size(); //输入的样本数
    cv::Mat distance = cv::Mat::zeros(cv::Size(1, num), CV_32FC1); //表示所有样本到当前聚类中心的距离
    cv::Mat minDistance = cv::Mat::zeros(cv::Size(1, num), CV_32FC1); //取较小距离
  
    cv::Mat classes = cv::Mat::zeros(cv::Size(1, num), CV_32SC1);     //表示类别
    center_index.push_back(index); //保存第一个聚类中心
    
    for (size_t i = 0; i < num; i++) {
      cv::Point2d diff = pts[i] - pts[center_index[0]];
      float d = sqrt(diff.x * diff.x + diff.y * diff.y);
      distance.at<float>(i, 0) = d;
      classes.at<int>(i, 0) = k + 1;
      if (maxDistance < d) {
        maxDistance = d;
        index = i; //与第一个聚类中心距离最大的样本
      }
    }
  
    minDistance = distance.clone();
    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;
    maxVal = maxDistance;
    while (maxVal > threshold) {
      k = k + 1;
      center_index.push_back(index); //新的聚类中心
      for (size_t i = 0; i < num; i++) {
        cv::Point2d diff = pts[i] - pts[center_index[k]];
        float d = sqrt(diff.x * diff.x + diff.y * diff.y);
        distance.at<float>(i, 0) = d;
        //按照当前最近临方式分类，哪个近就分哪个类别
        if (minDistance.at<float>(i, 0) > distance.at<float>(i, 0)) {
          minDistance.at<float>(i, 0) = distance.at<float>(i, 0);
          classes.at<int>(i, 0) = k + 1;
        }
      }
      //查找minDistance中最大值
      cv::minMaxLoc(minDistance, &minVal, &maxVal, &minLoc, &maxLoc);
      index = maxLoc.y;
    }
    return classes;
  }

int main( int argc, char** argv ) {
  // if( argc != 2) {
  //     std::cout <<" Usage: display_image ImageToLoadAndDisplay" << std::endl;
  //     return -1;
  // }

  // cv::Mat img;
  // img = cv::imread(argv[1], cv::IMREAD_COLOR);   // Read the file

  std::vector<cv::Point2d> pts;
  pts.push_back(cv::Point2d(1, 1));
  pts.push_back(cv::Point2d(10, 1));
  pts.push_back(cv::Point2d(500, 1));
  pts.push_back(cv::Point2d(500, 5));
  pts.push_back(cv::Point2d(502, 10));
  pts.push_back(cv::Point2d(490, 20));
  pts.push_back(cv::Point2d(490, 400));
  pts.push_back(cv::Point2d(490, 380));
  float threshold = 50;
  std::vector<int> center_index;
  cv::Mat classes = max_min_dist_clustering(pts, threshold, center_index);
  std::cout << classes << std::endl;
 
  
  // cv::Mat undist_img;
  // cv::imwrite("undist.jpg", undist_img);

  return 0;
}


