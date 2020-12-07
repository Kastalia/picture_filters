#include <opencv2/opencv.hpp>
#include <iostream>


void filter_linear_bgr(const uint8_t* h_input, uint8_t* h_output, size_t n_row, size_t n_col);
void filter_median_hsv(const uint8_t* h_input, uint8_t* h_output, size_t n_row, size_t n_col);
void calculate_histogram_brightness_hsv(const uint8_t* h_input, uint32_t* h_histogram, size_t n_row, size_t n_col);

int main() {
  cv::Mat pic = cv::imread("../partyhard.jpg",cv::IMREAD_COLOR);
  cv::imshow("partyhard_original", pic);
  cv::Mat pic_hsv;
  cvtColor(pic, pic_hsv,CV_BGR2HSV);
  cv::Mat pic_blur, pic_blur_hsv;
  clock_t start_s, stop_s;


  // linear filter
  pic_blur = cv::Mat::zeros(cv::Size(pic.cols, pic.rows), CV_8UC3);

  start_s = clock();
  filter_linear_bgr(pic.data, pic_blur.data, pic.rows, pic.cols);
  stop_s = clock();
  std::cout << "Time for the GPU blur linear filter: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << std::endl;

  cv::imshow("partyhard_filter_linear", pic_blur);


  //median filter
  pic_blur = cv::Mat::zeros(cv::Size(pic_hsv.cols, pic_hsv.rows), CV_8UC3);
  pic_blur_hsv = cv::Mat::zeros(cv::Size(pic_hsv.cols, pic_hsv.rows), CV_8UC3);

  start_s = clock();
  filter_median_hsv(pic_hsv.data, pic_blur_hsv.data, pic_hsv.rows, pic_hsv.cols);
  stop_s = clock();
  std::cout << "Time for the GPU blur median filter: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << std::endl;

  cvtColor(pic_blur_hsv, pic_blur,CV_HSV2BGR);
  cv::imshow("partyhard_filter_median", pic_blur);


  // histogram brightness. Построил по значению brightness в HSV представлении изображения.
  uint32_t hist_brightness[256];
  memset(hist_brightness, 0 , sizeof(uint32_t)*256);

  start_s = clock();
  calculate_histogram_brightness_hsv(pic_hsv.data, hist_brightness, pic_hsv.rows, pic_hsv.cols);
  stop_s = clock();
  std::cout << "Time for the GPU calculate histogram brightness: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << std::endl;

  int32_t sum_pixels=0;
  for(auto& elem:hist_brightness){
	std::cout<<elem<<" "<<std::endl;
	sum_pixels+=elem;
  }
  std::cout<<"Сумма="<<sum_pixels<<std::endl;
  cv::waitKey();
}