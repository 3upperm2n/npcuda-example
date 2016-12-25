import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

import time

npoints = 10000000
nfeatures = 10 


###----------------------------------------------------------------------------
### measure kernel compilation time
###----------------------------------------------------------------------------
start = time.time() 

mod = SourceModule("""
// 16K floats
__device__ __constant__ float clusters_cnst[16384];

__global__ void kernel_kmeans(const float* __restrict__ data,
		const int* __restrict__ membership,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		const int warps_per_blk,
		float* delta,
		int* new_membership,
		float *new_clusters,
		float *new_clusters_members)
{
	// assume 32 warps = 1024 block size
	__shared__ float warp_sm[32];
	__shared__ float feat_sm[32];
	__shared__ float change_sm[32];

	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	uint lx = threadIdx.x;

	float dist_min = 3.40282347e+38;
	int prev_id, curr_id;
	size_t data_base_ind = gx * nfeatures;
	size_t center_base_ind;
	
	int my_membership = -1;
	float change = 0.f;

	if(gx < npoints) {

		// load the membership
		my_membership = curr_id = prev_id = membership[gx];

		// go through each cluster
		for(int k=0; k<nclusters; k++)
		{
			float dist_cluster = 0.f;

			center_base_ind = k * nfeatures;

			// each feature dim
			for(int f=0; f<nfeatures; f++)
			{
				// fixme: data is frequently loaded
				float diff = data[data_base_ind + f] - clusters_cnst[center_base_ind + f];
				dist_cluster += diff * diff;
			}

			// update the id for the closest center
			if(dist_cluster < dist_min) {                                       
				dist_min = dist_cluster;                                        
				curr_id = k;                                                         
			}  
		}

		//--------------------------------------------------------------------//
		// update membership
		//--------------------------------------------------------------------//
		if(prev_id != curr_id) {
			my_membership = curr_id;
			new_membership[gx] = curr_id;	// update
			change = 1.f;
		}
	}


	int lane_id = threadIdx.x & 0x1F;
	int warp_id = threadIdx.x>>5;


	//---------------------------------------------------------------//
	// update delta
	//---------------------------------------------------------------//
	#pragma unroll                                                      
	for (int i=16; i>0; i>>=1) change += __shfl_down(change, i, 32);
	if(lane_id == 0) change_sm[warp_id] = change;
	__syncthreads();
	if(warp_id == 0) {
		change = (lx < warps_per_blk) ? change_sm[lx] : 0;	
		#pragma unroll
		for (int i=16; i>0; i>>=1) change += __shfl_down(change, i, 32);
		if(lx == 0) {
			atomicAdd(&delta[0], change);	
		}
	}


	for(int k=0; k<nclusters; k++)
	{
		int   flag = 0;
		float tmp  = 0.f;			// for membership number
		if(my_membership == k) {
			flag = 1;
			tmp  = 1.f;	
		}

		//-------------------------------------------------------------//
		// counter the members for current cluster 
		//-------------------------------------------------------------//
		// warp reduction 
#pragma unroll                                                      
		for (int i=16; i>0; i>>=1 ) {                                       
			tmp += __shfl_down(tmp, i, 32);
		}

		if(lane_id == 0) {
			warp_sm[warp_id] = tmp;
		}

		__syncthreads();

		if(warp_id == 0) {
			tmp = (lx < warps_per_blk) ? warp_sm[lx] : 0.f;	

#pragma unroll
			for (int i=16; i>0; i>>=1) {                                       
				tmp += __shfl_down(tmp, i, 32);
			} 

			if(lx == 0) { 	// add the local count to the global count
				atomicAdd(&new_clusters_members[k], tmp);	
			}
		}



		//------------------------------------------------------------//
		// accumuate new clusters for each feature dim
		//------------------------------------------------------------//
		float feat;
		for(int f=0; f<nfeatures; f++) 
		{
			
			// load feature value for current data point
			feat = 0.f;
			if(flag == 1) {
				feat = data[gx * nfeatures + f];
			}

			//------------------------------------//
			// reduction for feature values
			//------------------------------------//
			// sum current warp
			#pragma unroll                                                      
			for (int i=16; i>0; i>>=1) feat += __shfl_down(feat, i, 32);
			// save warp sum to shared memory using the 1st lane
			if(lane_id == 0) feat_sm[warp_id] = feat;
			__syncthreads();
			// use the 1st warp to accumulate the block sum
			if(warp_id == 0) {
				feat = (lx < warps_per_blk) ? feat_sm[lx] : 0.f;	
				#pragma unroll
				for (int i=16; i>0; i>>=1) feat += __shfl_down(feat, i, 32);
				// add the local count to the global count
				if(lx == 0) {
					atomicAdd(&new_clusters[k * nfeatures + f], feat);	
				}
			}
		}

	}
}

""")


end = time.time() 

print 'kernel compilation time : ' + str(end - start) + ' s'


###----------------------------------------------------------------------------
### end of measure kernel compilation time
###----------------------------------------------------------------------------


### ------------------------------------------
### start timing the start of the end-to-end processing time 
### ------------------------------------------
start = time.time()


# link to the kernel function
kernel_kmeans = mod.get_function('kernel_kmeans')

# constant memory
clusters_cnst = mod.get_global('clusters_cnst')[0]


#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
tol = 0.0001
nclusters = 2
maxiters = 300

# input data
X = np.random.rand(npoints, nfeatures).astype('f')
#X

# init centroids
clusters = X[:nclusters,]

# init membership to zeros
membership = np.zeros(npoints).astype('i')

# delta
delta = np.zeros(1).astype('f')


###
## allocate memory on device
###
X_gpu = cuda.mem_alloc(X.nbytes)

membership_gpu     = cuda.mem_alloc(membership.nbytes)

new_membership_gpu = cuda.mem_alloc(membership.nbytes)

# new clusters
new_clusters = np.zeros(clusters.shape).astype('f')
new_clusters_gpu = cuda.mem_alloc(clusters.nbytes)

# new clusters membership
new_clusters_member = np.zeros(nclusters).astype('f')
new_clusters_member_gpu = cuda.mem_alloc(new_clusters_member.nbytes)

delta_gpu = cuda.mem_alloc(delta.nbytes)

###
## transfer data to gpu
###
cuda.memcpy_htod(X_gpu, X)

cuda.memcpy_htod(membership_gpu, membership)

# The kernel goes into constant memory via a symbol defined in the kernel
cuda.memcpy_htod(clusters_cnst,  clusters)

###
## define kernel configuration
###
blk_size = 128
grd_size = (npoints + blk_size -1) / blk_size
warps_per_blk = 128 / 32


###---------------------------------------------------------------------------
### Run kmeans on gpu
###---------------------------------------------------------------------------
loop_count = 0

for i in range(0,maxiters):
    
    loop_count = loop_count + 1
    
    # change to zero for each iteration
    delta = np.zeros(1).astype('f')
    cuda.memcpy_htod(delta_gpu, delta)
    
    # start from zero for each iteration
    new_clusters = np.zeros(clusters.shape).astype('f')
    new_clusters_member = np.zeros(nclusters).astype('f')
    
    # copy 
    cuda.memcpy_htod(new_clusters_gpu, new_clusters)
    cuda.memcpy_htod(new_clusters_member_gpu, new_clusters_member)
    
    
    
    ## run kernel
    kernel_kmeans(X_gpu, membership_gpu,\
              np.int32(npoints), np.int32(nfeatures), np.int32(nclusters), np.int32(warps_per_blk), \
              delta_gpu, \
              new_membership_gpu, \
                new_clusters_gpu, 
                  new_clusters_member_gpu, \
                  block = (blk_size, 1, 1), grid = (grd_size, 1, 1))

    # copy back
    cuda.memcpy_dtoh(delta, delta_gpu)
    
    if(delta[0] < tol):
        break
        
    # copy back new_clusters
    cuda.memcpy_dtoh(new_clusters, new_clusters_gpu)
    cuda.memcpy_dtoh(new_clusters_member, new_clusters_member_gpu)
    
    ## re-compute clusters
    for i in xrange(0, clusters.shape[0]):
        for j in xrange(0, clusters.shape[1]):
            clusters[i,j] = new_clusters[i,j] / new_clusters_member[i]
            
    
    ## copy to constant memory
    cuda.memcpy_htod(clusters_cnst,  clusters)
    
    # update membership
    cuda.memcpy_dtod(membership_gpu, new_membership_gpu, membership.nbytes)


###----------------------------------------------------------------------------
## end of gpu kmeans
###----------------------------------------------------------------------------

# copy back new_clusters
cuda.memcpy_dtoh(new_clusters, new_clusters_gpu)
cuda.memcpy_dtoh(new_clusters_member, new_clusters_member_gpu)


### output
## re-compute clusters
for i in xrange(0, clusters.shape[0]):
    for j in xrange(0, clusters.shape[1]):
        clusters[i,j] = new_clusters[i,j] / new_clusters_member[i]
        

### output
cuda.memcpy_dtoh(membership, new_membership_gpu)



### ------------------------------------------
### end timing of the end-to-end processing time 
### ------------------------------------------
end = time.time()
runtime = end - start


iter_time = runtime / loop_count
###----------------------------------------------------------------------------
## dump stat 
###----------------------------------------------------------------------------

print 'runtime : ' + str(runtime)  + ' s'                                                  
print 'runtime per iter : ' + str(iter_time) + ' s'                             
print 'niter : ' + str(loop_count)                                                   
print 'maxiter : ' + str(maxiters) 
