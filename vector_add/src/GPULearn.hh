class GPULearn
{
public:
	// host
	float* a_h;
	float* b_h;
	float* c_h;

	// device 
	float* a_d;
	float* b_d;
	float* c_d;

	int length;

	GPULearn(float* A, int DIM1, float* B, int DIM2);

	~GPULearn();

	int BLK(int number, int blocksize) {
		return (number + blocksize - 1) / blocksize;	
	}

	void vectorAdd();

	void getData();

	void getData_extern(float *c_out, int dim_c);
};
