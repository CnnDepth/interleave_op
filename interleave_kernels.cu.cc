//#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "interleave.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <cuda/include/cuda.h>

namespace tensorflow{
namespace functor{

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void InterleaveCudaKernel( const int out_size, const int N, const int H, const int W, const int C, const T* in1, const T* in2, const T* in3, const T* in4, T* out) 
{	
  //int index = blockIdx.x * blockDim.x + threadIdx.x;	
  
  for( int index = blockIdx.x * blockDim.x + threadIdx.x; index < out_size; index += blockDim.x * gridDim.x )
  {
	int n = index % N;   
	int h = ( ( index - n ) / N ) % H;
	int w = ( ( index - h * N - n ) / (N * H) ) % W;
	int c = ( ( index - w * H * N - h * N - n) / ( N * H * W ) ) % C; 
	
	int is_h_even = h % 2;
	int is_w_even = w % 2;
	
	if( !is_h_even )
	{
		if( !is_w_even )
		{
			out[index] = in1[n + ( ( h / 2 ) ) * N + ( ( w / W ) ) * N * H / 2 + c * N * H * W / 4];
		}
		else
		{
			out[index] = in3[ n + ( ( h / 2 ) ) * N + ( ( w / W ) ) * N * H / 2 + c * N * H * W / 4];
		}
	}
	else
	{
		if( !is_w_even )
		{
			out[index] = in2[ n + ( ( h / 2 ) ) * N + ( ( w / W ) ) * N * H / 2 + c * N * H * W / 4  ];
		}
		else
		{
			out[index] = in4[ n + ( ( h / 2 ) ) * N + ( ( w / W ) ) * N * H / 2 + c * N * H * W / 4  ];
		}
		
	}
  }  
}  

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct InterleaveFunctor<GPUDevice, T>{ 
  void operator()(const GPUDevice& d, const int size, const shape_t& target_shape, const T* in1, const T* in2, const T* in3, const T* in4, T* out) 
  {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  
  InterleaveCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>( size, target_shape.n, target_shape.h, target_shape.w, target_shape.c, in1, in2, in3, in4, out);
  }
};
// Explicitly instantiate functors for the types of OpKernels registered.
template struct InterleaveFunctor<GPUDevice, float>;
template struct InterleaveFunctor<GPUDevice, int32>;
template struct InterleaveFunctor<GPUDevice, double>;
}//namespace functor
}//namespace tensorflow

//#endif  // GOOGLE_CUDA

