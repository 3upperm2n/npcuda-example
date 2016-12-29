import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import compile

test_kernel = compile(
"""
__global__ void kernel_sigmoid (const float* __restrict__ d_xt,
		const int* __restrict__ d_y,
		const int input_size,
		float *d_sigmoid_error)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);	
	if(gx < input_size) {
		float z = -d_xt[gx];
		float h = 1.f / ( 1.f + expf(z) );
		d_sigmoid_error[gx] = d_y[gx] - h;
	}
}

__global__ void kernel_update_weight (const float* __restrict__ d_theta_tmp, 
		const float alpha,
		const int feature_size,
		float *d_theta)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);	
	if(gx < feature_size) {
		float weight = d_theta[gx];
		d_theta[gx]  = weight + d_theta_tmp[gx] * alpha;
	}
}
""")

with open("logisticReg_kernels.cubin", "wb") as file:
    file.write(test_kernel)


