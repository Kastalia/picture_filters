#include <opencv2/opencv.hpp>
#include <iostream>


void filter_linear(const uint8_t* h_input, uint8_t* h_output, size_t n_row, size_t n_col);

int main() {
  std::cout<<"its cpu time\n";

  cv::Mat pic = cv::imread("../partyhard.jpg",cv::IMREAD_COLOR);
  cv::Mat pic_blur = cv::Mat::zeros(cv::Size(pic.cols, pic.rows), CV_8UC3);

  clock_t start_s = clock();
  filter_linear(pic.data, pic_blur.data, pic.rows, pic.cols);
  clock_t stop_s = clock();
  std::cout << "Time for the GPU blur linear filter: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << std::endl;

  cv::imshow("partyhard_original", pic);
  cv::imshow("partyhard_filter_linear", pic_blur);
  cv::waitKey();

}