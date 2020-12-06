#include <iostream>
#include <opencv2/opencv.hpp>

// kernel
__global__ void calculate_histogram_brightness_hsv_kernel(const uint8_t* picture, uint32_t* histogram_brightness) {
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;

  uint8_t brightness = picture[(i*blockDim.x+j)*3+2];
  histogram_brightness[brightness] += 1;
  __syncthreads();
}



__host__ void calculate_histogram_brightness_hsv(const uint8_t* h_input, uint32_t* h_histogram, size_t n_row, size_t n_col) {
  size_t size_input = sizeof(uint8_t) * n_row * n_col * 3;
  size_t size_output = sizeof(uint32_t)*256;
  uint8_t* d_input;
  cudaMalloc(&d_input, size_input);
  cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);

  uint32_t* d_output;
  cudaMalloc(&d_output, size_output);
  cudaMemset(d_output, 0, size_output);

  calculate_histogram_brightness_hsv_kernel<<<
  dim3(n_row,1,1),
  dim3(n_col, 1, 1)
  >>>(d_input, d_output);

  cudaMemcpy(h_histogram, d_output, size_output, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_input);
  cudaFree(d_output);
}