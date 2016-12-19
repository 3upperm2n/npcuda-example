#include <stdio.h>

void __global__ kernel_vectorAdd(const float* __restrict__ a,  
		                         const float* __restrict__ b, 
						         float* c, 
						         int length) 
{
    uint gid = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if(gid < length) 
	{
		c[gid] = a[gid] + b[gid];
    }
}
