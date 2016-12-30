#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <time.h>                                                               
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <kernel.cu>
#include <kmeans.hh>

//using namespace std;

// input from python:
// 		data, nclusters, tol (threshold), maxiters
// output:
//		labels (membership), iterations, centroids (clusters) 


KmeansGPU::KmeansGPU(float threshold_, 
		             int cluster_num_, 
					 int npoints_, 
		             int nfeatures_,
		             int maxiter,
					 float* data_in
					 )
{
	tic();

	threshold   = threshold_;
	nclusters 	= cluster_num_;
	npoints 	= npoints_;
	nfeatures 	= nfeatures_;
	nloops 		= maxiter;


	membership_bytes 	= npoints * sizeof(int);
	clusters_bytes 		= nclusters * nfeatures * sizeof(float);
	data_bytes 			= npoints * nfeatures * sizeof(float);
	clusters_members_bytes 		= nclusters * sizeof(float);

	// read data to unified memory	
	if(data != NULL) cudaFree(data);
	cudaMallocManaged((void **)&data, data_bytes);
	cudaMemcpy(data, data_in, data_bytes, cudaMemcpyHostToDevice);

	// allocate membership
	if(membership != NULL) cudaFree(membership);
	if(new_membership != NULL) cudaFree(new_membership);
	cudaMallocManaged((void**)&membership,     membership_bytes);
	cudaMallocManaged((void**)&new_membership,     membership_bytes);

	// allocate delta
	if(delta != NULL) cudaFree(delta);
	cudaMallocManaged((void**)&delta,     sizeof(float));

	// clusters, new clusters
	if(clusters != NULL) cudaFree(clusters);
	if(new_clusters != NULL) cudaFree(new_clusters);
	if(new_clusters_members != NULL) cudaFree(new_clusters_members);

	cudaMallocManaged((void**)&clusters,             clusters_bytes);
	cudaMallocManaged((void**)&new_clusters,         clusters_bytes);
	cudaMallocManaged((void**)&new_clusters_members, clusters_members_bytes);

	//--------------------------------------------------------------------//
	// pick the first [nclusters] samples as the initial clusters           
	//--------------------------------------------------------------------//
	for(int i=0; i<nclusters; i++) {                                        
		for(int j=0; j<nfeatures; j++) {                                    
			clusters[i * nfeatures + j] = data_in[i * nfeatures + j];           
		}
	}

	// gpu kernel configuration
	blocksize = 128;
	warps_per_blk = blocksize >> 5;


	blkDim = dim3(blocksize, 1, 1);
	grdDim = dim3(BLK(npoints, blocksize), 1, 1);

	/*
	printf("\ndata_in\n");
	print_array(data_in, npoints, nfeatures);

	printf("\nclusters\n");
	print_array(clusters, nclusters, nfeatures);

	printf("threshold : %f\n", threshold);
	printf("nclusters: %d\n", nclusters);
	printf("nfeatures: %d\n", nfeatures);
	printf("npoints: %d\n", npoints);
	printf("maxiter: %d\n", nloops);

	printf("warps per blk : %d\n", warps_per_blk);
	printf("grd : %d, blk : %d\n", grdDim.x, blkDim.x);
	*/

	toc("construct");
}

KmeansGPU::~KmeansGPU() {
	tic();
	Cleanup();
	toc("de-construct");
}

void KmeansGPU::print_array(float *array, int row, int col) {

	for(int i=0; i<row; i++) {                                        
		int startpos = i * col;
		for(int j=0; j<col; j++) {                                    
			printf("%f ", array[startpos + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void KmeansGPU::print_array(int *array, int row, int col) {

	for(int i=0; i<row; i++) {                                        
		int startpos = i * col;
		for(int j=0; j<col; j++) {                                    
			printf("%d ", array[startpos + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void KmeansGPU::Cleanup() {
	if(data       != NULL)                 cudaFree(data);
	if(membership != NULL)                 cudaFree(membership);
	if(new_membership != NULL)             cudaFree(new_membership);
	if(delta      != NULL)                 cudaFree(delta);
	if(clusters   != NULL)                 cudaFree(clusters);
	if(new_clusters != NULL)               cudaFree(new_clusters);
	if(new_clusters_members != NULL)       cudaFree(new_clusters_members);
}



//----------------------------------------------------------------------------//
// Run Kmeans
//----------------------------------------------------------------------------//
void KmeansGPU::Run()                                                
{
	tic();

	if (nclusters > npoints) {                                              
		fprintf(stderr, "Can't have more clusters (%d) than the points (%d)!\n",    
				nclusters, npoints);                                        
		Cleanup();
		exit(1);                                                            
	}                                                                       

	//----------------------//
	// copy clusters to contant memory
	//----------------------//
	cudaMemcpyToSymbol(clusters_cnst, clusters, clusters_bytes, 0, cudaMemcpyHostToDevice);

	// the membership is intialized with 0 
	cudaMemset(membership, 0, membership_bytes);


	loop_count = 0;

	int cnt = 1;
	for(int i=0; i<nloops; i++)
	{
		cnt = Kmeans_gpu();
		if(cnt == 0) break;
	}

	//printf("loop count : %d\n", loop_count);

	toc("run");
}


//----------------------------------------------------------------------------//
// Run Kmeans : GPU Kernels
//----------------------------------------------------------------------------//
int KmeansGPU::Kmeans_gpu()
{
	loop_count++;

	// change to 0 for each iteration
	delta[0] = 0.f;

	// start from zero for each iteration
	cudaMemset(new_clusters,         0, clusters_bytes);
	cudaMemset(new_clusters_members, 0, clusters_members_bytes);

	/*
	printf("\nnew clusters\n");
	print_array(clusters, nclusters, nfeatures);

	printf("\nnew cluster member\n");
	print_array(new_clusters_members, nclusters, 1);
	*/

	//cudaDeviceSynchronize();

	// run gpu kernel
	kernel_kmeans <<< grdDim, blkDim >>> (data, 
			membership, 
			npoints, 
			nfeatures, 
			nclusters, 
			warps_per_blk,
			delta,
			new_membership,
			new_clusters,
			new_clusters_members);

	cudaDeviceSynchronize();

	/*
	printf("\nnew clusters\n");
	print_array(clusters, nclusters, nfeatures);

	printf("\nnew cluster member\n");
	print_array(new_clusters_members, nclusters, 1);

	//printf("\nnew membership \n");
	//print_array(new_membership, npoints, 1);

	printf("\ndelta\n");
	print_array(delta, 1, 1);
	*/


	// update the clusters on the host/cpu 
	for(int k=0; k<nclusters; k++) {                                        
		int startpos = k * nfeatures;
		for(int f=0; f<nfeatures; f++) {                                    
			clusters[startpos + f] = new_clusters[startpos + f] / new_clusters_members[k];           
		}                                                                   
	}

	/*
	printf("\nupdated clusters\n");
	print_array(clusters, nclusters, nfeatures);
	*/

	// check the termination condition
	if(delta[0] < threshold) {
		return 0;
	}

	// update clusters in the constant memsory 
	cudaMemcpyToSymbol(clusters_cnst, clusters, clusters_bytes, 0, cudaMemcpyHostToDevice);

	// update membership
	cudaMemcpy(membership, new_membership, membership_bytes, cudaMemcpyDeviceToDevice);

	return 1;
}

void KmeansGPU::getData_extern(int *membership_out, int &iterations_out, float *centroids_out)
{
	tic();

	//printf("loop count : %d\n", loop_count);

	//printf("\noutput clusters\n");
	//print_array(clusters, nclusters, nfeatures);

	

	// copy new_membership (on the cpu) to the output
	memcpy(membership_out, new_membership, membership_bytes);

	// update iterations
	iterations_out = loop_count;

	// copy clusters (on the cpu) to the output

	memcpy(centroids_out, clusters, clusters_bytes);

	toc("pass data");
}

