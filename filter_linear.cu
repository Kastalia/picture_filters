#include <iostream>
#include <opencv2/opencv.hpp>

// kernel
__global__ void filter_linear_apply(uint8_t* picture, uint8_t* picture_blur) {
  size_t sum_filter = 0;
  uint8_t filter[25];
  for (auto &f : filter) {
	f = 1;
	sum_filter += f;
  }
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  uint8_t partpic_blue;
  uint8_t partpic_green;
  uint8_t partpic_red;
  size_t pos_pixel = 0;

  //picture_blur[3 * i * blockDim.x + 3 * j] = 255;
  //picture_blur[3 * i * blockDim.x + 3 * j + 1] = 255;
  //picture_blur[3 * i * blockDim.x + 3 * j + 2] = 0;
  //printf("%d ",picture[3 * i * blockDim.x + 3 * j + 2]);

  //printf("gDim.x_rows_=%d  bDim.x_columns_=%d bIdx.x_i_=%d  tIdx.x_j_=%d pic_blue=%d filter_blue=%d\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, picture[3 * i * blockDim.x + 3 * j], picture_blur[3 * i * blockDim.x + 3 * j]);

  for(int k=-2;k<=2; k++)
  {
	for(int l=-2; l<=2; l++)
	{
	  if(((i+k+1)>gridDim.x)|((i+k)<0)|((j+l+1)>blockDim.x)|((j+l)<0)) {
		//вышли за рамки, будем усредняться по преобразуемому пикселю
		pos_pixel = i * blockDim.x + j;
		partpic_blue = picture[pos_pixel*3];
		partpic_green = picture[pos_pixel*3+1];
		partpic_red = picture[pos_pixel*3+2];
	  }
	  else {
		pos_pixel = (i + k) * blockDim.x + j + l;
		partpic_blue = picture[pos_pixel*3];
		partpic_green= picture[pos_pixel*3+1];
		partpic_red= picture[pos_pixel*3+2];
	  }
	  pos_pixel = i*blockDim.x+j;
	  picture_blur[pos_pixel*3]+=1.0/sum_filter*filter[(k+2)*5+l+2]*partpic_blue;
	  picture_blur[pos_pixel*3+1]+=1.0/sum_filter*filter[(k+2)*5+l+2]*partpic_green;
	  picture_blur[pos_pixel*3+2]+=1.0/sum_filter*filter[(k+2)*5+l+2]*partpic_red;
	};
  };
  /*
   * GUIDE:
  for(int k=-2;k<=2; k++)
  {
	for(int l=-2; l<=2; l++)
	{
	  u[i][j]+=1/s*a[k][l]*u[i+k][j+l];
	}
  }
  cudaDeviceSynchronize();
  */
  __syncthreads();

}



__host__ void filter_linear(const uint8_t* h_input, uint8_t* h_output, size_t n_row, size_t n_col) {
  size_t size = sizeof(uint8_t) * n_row * n_col * 3;
  uint8_t* d_input;
  cudaMalloc(&d_input, size);
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  uint8_t* d_output;
  cudaMalloc(&d_output, size);
  cudaMemset(d_output, 0, size);

  filter_linear_apply<<<
  dim3(n_row,1,1),
  dim3(n_col, 1, 1)
  >>>(d_input, d_output);

  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_input);
  cudaFree(d_output);
}