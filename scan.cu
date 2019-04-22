#include <stdio.h>

__global__ void scan_inclusion(float * d_in,float * d_out){
	//index = threadIdx.x + (blockDim.x * blockIdx.x);
	int tid   = threadIdx.x;

	extern __shared__ float  sdata[10];

	sdata[tid] = d_in[tid];
	__syncthreads();

	float temp;
	for (int s=1;s<blockDim.x/2;s=s<<1) {
		
		if(tid>=s) {
			temp = sdata[tid]+sdata[tid-s];
			__syncthreads();
			sdata[tid] = temp;
		}	
		__syncthreads();
	}
       d_out[tid] = sdata[tid];	
}

int main(){
	const int  maxthreadsinablock = 10;
        const int THREADS= 10;
	const int BLOCKS = 1;
	const int ARRAY_SIZE = 10;

	//input array on the host
	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];

	//Generate input array for testing
	for (int i=0;i<ARRAY_SIZE;i++) {
		h_in[i] = float(i);
	}
	//declare GPU memory pointers
	float *d_in;
	float *d_out;

	//allocate memory on GPU
	cudaMalloc((void **) &d_in,sizeof(float)*10);
	cudaMalloc((void **) &d_out,sizeof(float)*10);

	cudaMemcpy(d_in, h_in, sizeof(float)*10,cudaMemcpyHostToDevice);

	scan_inclusion<<<BLOCKS,THREADS>>>(d_in,d_out);

	cudaMemcpy(h_out, d_out, sizeof(float)*10,cudaMemcpyDeviceToHost);


	//printing results
	for (int j=0;j<ARRAY_SIZE;j++) {
		printf ("%f\t",h_out[j]);
	}
	cudaFree(d_in);
	cudaFree(d_out);

return 0;
}

	
