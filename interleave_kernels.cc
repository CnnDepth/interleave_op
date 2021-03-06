#include "interleave.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow{


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor{
// CPU specialization of actual computation.
template <typename T>
struct InterleaveFunctor<CPUDevice, T> 
{
  void operator()(const CPUDevice& d, const int out_size, const shape_t& target_shape, const T* in1, const T* in2, const T* in3, const T* in4, T* out) 
  {
	int N = target_shape.n;
	int H = target_shape.h;
	int W = target_shape.w;
	int C = target_shape.c;  
	  
    for( int index = 0; index < out_size; ++index )
    {
		int n = index / (H * W * C);   
		int h = (index % (H * W * C)) / (W * C);
		int w = (index % (W * C)) / C;
		int c = index % C; 
	
		int is_h_even = h % 2;
		int is_w_even = w % 2;

    int index_in_input = n * H * W * C / 4 + (h / 2) * W * C / 2 + (w / 2) * C + c;
	
		if( !is_h_even )
		{
			if( !is_w_even )
			{
				out[index] = in1[index_in_input];
			}
			else
			{
				out[index] = in2[index_in_input];
			}
		}
		else
		{
			if( !is_w_even )
			{
				out[index] = in3[index_in_input];
			}
			else
			{
				out[index] = in4[index_in_input];
			}
		}
		  
	  }
	}
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class InterleaveOp : public OpKernel {
 public:
  explicit InterleaveOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES( context, context->num_inputs() == 4,
                 errors::InvalidArgument("Interleave expects 4 inputs") );
    // Grab the input tensor
    
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    const Tensor& input_tensor3 = context->input(2);
    const Tensor& input_tensor4 = context->input(3);
     
    //Check if the input is 4-d tensor  
    OP_REQUIRES( context, input_tensor1.shape().dims() == 4,
                 errors::InvalidArgument("Interleave expects 4-d tensor") );
    OP_REQUIRES( context, input_tensor2.shape().dims() == 4,
                 errors::InvalidArgument("Interleave expects 4-d tensor") );
    OP_REQUIRES( context, input_tensor3.shape().dims() == 4,
                 errors::InvalidArgument("Interleave expects 4-d tensor") );
    OP_REQUIRES( context, input_tensor4.shape().dims() == 4,
                 errors::InvalidArgument("Interleave expects 4-d tensor") );

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape out_shape = input_tensor1.shape();
    //changing h and w to become in_h * 2, in_w * 2
    out_shape.set_dim(1, input_tensor1.shape().dim_size(1) * 2);
    out_shape.set_dim(2, input_tensor1.shape().dim_size(2) * 2);
    
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor1.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
	
	shape_t target_shape = { static_cast<int>( output_tensor->shape().dim_size(0) )
					       , static_cast<int>( output_tensor->shape().dim_size(1) )
	 				       , static_cast<int>( output_tensor->shape().dim_size(2) )
	    				   , static_cast<int>( output_tensor->shape().dim_size(3) )};
	
	//std::cout << "shape: " << target_shape.n << " " << target_shape.h << std::endl;
	
    InterleaveFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(output_tensor->NumElements()),
        target_shape, 
        input_tensor1.flat<T>().data(),
        input_tensor2.flat<T>().data(),
        input_tensor3.flat<T>().data(),
        input_tensor4.flat<T>().data(),
        output_tensor->flat<T>().data());
    //std::cout << "Kernel called: " << context->eigen_device<Device>()
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Interleave").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      InterleaveOp<CPUDevice, T>);

//REGISTER_CPU(float);
//REGISTER_CPU(int32);
//REGISTER_CPU(double);

// Register the GPU kernels.

#define REGISTER_GPU(T) \
  extern template struct InterleaveFunctor<GPUDevice, T>; \
  REGISTER_KERNEL_BUILDER( \
      Name("Interleave").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      InterleaveOp<GPUDevice, T>);

REGISTER_GPU(float);
REGISTER_GPU(int32);
REGISTER_GPU(double);
//#endif  // GOOGLE_CUDA
}//namespace functor
}//namespace tensorflow
