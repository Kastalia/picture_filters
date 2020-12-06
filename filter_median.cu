#include <iostream>
#include <opencv2/opencv.hpp>

__device__ uint8_t median_pixel(uint8_t *picture, size_t n_row_window, size_t n_col_window, size_t n_col_picture) {
  uint32_t counting[256];
  memset(counting, 0 ,sizeof(uint32_t)*256);
  uint8_t pixel_picture=0;
  for(int i=0;i<n_row_window;++i)
	for (int j=0;j<n_col_window;++j){
	  pixel_picture = picture[(i*n_col_picture+j)*3];
	  counting[pixel_picture]+=1;
	}
  uint32_t median_border = (n_row_window*n_col_window-1)/2;
  uint32_t tmp=0;
  uint8_t pixel_median=-1;
  while(tmp<median_border){
	pixel_median+=1;
	tmp+=counting[pixel_median];
  }
  return pixel_median;
}

// kernel
__global__ void filter_median_apply_hsv(uint8_t* picture, uint8_t* picture_blur) {
  size_t median_col = 5;
  size_t median_row = 5;

  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  size_t window_upper = i-(median_row-1)/2;
  size_t window_lower = i+(median_row-1)/2;
  size_t window_left = j-(median_col-1)/2;
  size_t window_right = j+(median_col-1)/2;
  size_t pos_pixel = i*blockDim.x+j;

  if ((window_upper<0)
	  |((window_lower+1)>gridDim.x)
	  |(window_left<0)
	  |((window_right+1)>blockDim.x)){
	picture_blur[pos_pixel*3] = picture[pos_pixel*3];
	picture_blur[pos_pixel*3+1] = picture[pos_pixel*3+1];
	picture_blur[pos_pixel*3+2] = picture[pos_pixel*3+2];
  }
  picture_blur[pos_pixel*3] = median_pixel(&picture[(window_upper*blockDim.x+window_left)*3], median_row, median_col, blockDim.x);
  picture_blur[pos_pixel*3+1] = median_pixel(&picture[(window_upper*blockDim.x+window_left)*3+1], median_row, median_col, blockDim.x);
  picture_blur[pos_pixel*3+2] = median_pixel(&picture[(window_upper*blockDim.x+window_left)*3+2], median_row, median_col, blockDim.x);
  __syncthreads();
}

__host__ void filter_median_hsv(const uint8_t *h_input, uint8_t *h_output, size_t n_row, size_t n_col) {
  size_t size = sizeof(uint8_t) * n_row * n_col * 3;
  uint8_t *d_input;
  cudaMalloc(&d_input, size);
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  uint8_t *d_output;
  cudaMalloc(&d_output, size);
  cudaMemset(d_output, 0, size);

  filter_median_apply_hsv<<<
	  dim3(n_row, 1, 1),
	  dim3(n_col, 1, 1)>>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_input);
  cudaFree(d_output);
}