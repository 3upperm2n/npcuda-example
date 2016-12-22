#include <cuda_runtime.h>

class KmeansGPU
{
public:
	KmeansGPU(float threshold_, 
			int cluster_num_, 
			int npoints_, 
			int nfeatures_,
			int maxiter,
			float* data_in
			);

	~KmeansGPU();

	void Cleanup();

	int BLK(int num, int blksize) {
		return (num + blksize - 1) / blksize;	
	}

	void print_array(float *array, int row, int col);
	void print_array(int   *array, int row, int col);

	void Run();
	int Kmeans_gpu(); 	// return: continue or not
	void getData_extern(int*membership_out, int &iterations_out,float *centroids_out);


	float 			threshold;
	int 			nclusters;
	int 			npoints;
	int 			nfeatures;
	int 			nloops;

	// using unified memory
	float 			*data;
	int 			*membership;
	int 			*new_membership;
	float           *delta;
	float           *clusters;
	float           *new_clusters;
	float           *new_clusters_members;

	// size
	size_t membership_bytes;
	size_t clusters_bytes;
	size_t data_bytes;
	size_t clusters_members_bytes;
	
	int loop_count;

	// kernel configuration
	int blocksize;
	int warps_per_blk;

	dim3 blkDim;
	dim3 grdDim;
};
