#include <stdio.h>

// 16K floats
__constant__ float clusters_cnst[16384];

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
