// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

#include <tensorflow/core/util/tensor_format.h>

namespace tensorflow{
namespace functor{

  typedef struct shape
  {
	int n;
	int h;
	int w;
	int c;  
  }shape_t; 

template <typename Device, typename T>
struct InterleaveFunctor
{		
  void operator()(const Device& d, const int out_size, const shape_t&, const T* in1, const T* in2, const T* in3, const T* in4, T* out);
};

//#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
//template <typename Eigen::GpuDevice, typename T>
//#struct InterleaveFunctor {
//  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
//};
//#endif
}//functor
}//tensorflow

#endif //KERNEL_EXAMPLE_H_
