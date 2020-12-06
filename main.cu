#include <opencv2/opencv.hpp>
#include <iostream>


void filter_linear_bgr(const uint8_t* h_input, uint8_t* h_output, size_t n_row, size_t n_col);
void filter_median_hsv(const uint8_t* h_input, uint8_t* h_output, size_t n_row, size_t n_col);

int main() {
  std::cout<<"its cpu time\n";

  cv::Mat pic = cv::imread("../partyhard.jpg",cv::IMREAD_COLOR);

  // linear filter
  cv::Mat pic_blur = cv::Mat::zeros(cv::Size(pic.cols, pic.rows), CV_8UC3);
  /*
  clock_t start_s = clock();
  filter_linear_bgr(pic.data, pic_blur.data, pic.rows, pic.cols);
  clock_t stop_s = clock();
  std::cout << "Time for the GPU blur linear filter: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << std::endl;

  cv::imshow("partyhard_original", pic);
  cv::imshow("partyhard_filter_linear", pic_blur);
  cv::waitKey();
   */


  cv::Mat pic_hsv;
  cvtColor(pic, pic_hsv,CV_BGR2HSV);
  cv::Mat pic_blur_hsv = cv::Mat::zeros(cv::Size(pic_hsv.cols, pic_hsv.rows), CV_8UC3);
  pic_blur = cv::Mat::zeros(cv::Size(pic_hsv.cols, pic_hsv.rows), CV_8UC3);

  clock_t start_s = clock();
  filter_median_hsv(pic_hsv.data, pic_blur_hsv.data, pic_hsv.rows, pic_hsv.cols);
  clock_t stop_s = clock();
  std::cout << "Time for the GPU blur median filter: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << std::endl;

  cvtColor(pic_blur_hsv, pic_blur,CV_HSV2BGR);
  cv::imshow("partyhard_original", pic);
  cv::imshow("partyhard_filter_median", pic_blur);

  cv::waitKey();


}