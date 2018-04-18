8i#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<cuda.h>

typedef char* string;

__host__
void sum(float *A, float *B, float *C, int size){
	for (int i = 0; i < size; i++) C[i] = A[i] + B[i];
}

__global__
void sumCU(float *A, float *B, float *C, int size){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < size) C[tid] = A[tid] + B[tid];
}

__host__
bool compare(float *A, float *B, int size) {
	for (int i = 0; i < size; i++) if (A[i] != B[i]) return false;
	return true;
}

__host__
void print(float *M, int size) {
	printf("---------------print matrix--------------\n");
	for (int i = 0; i < size; i++) printf("%f ", M[i]);
	printf("\n");
}

__host__
void receive(float *M, FILE *stream, int size) {
	int i;
	for (i = 0; i < size; i++) fscanf(stream, "%f,", &M[i]);
	fclose(stream);
}

int main(int argc, char** argv){

	if (argc != 3) {
		printf("Must be called with the names of the files\n");
		return 1;
	}

	double timeCPU, timeGPU;
	time_t start,end;
	int sizeA, sizeB;

	// --------------- CPU ----------------
	cudaError_t error = cudaSuccess;
	float *h_A, *h_B, *h_C;

	FILE *f1, *f2;
	f1 = fopen(argv[1], "r");
	f2 = fopen(argv[2], "r");

	fscanf(f1, "%d", &sizeA);
	fscanf(f2, "%d", &sizeB);

	if (sizeA != sizeB) {
		printf("The matrix should have same dimensions\n");
		return 1;
	}

	float size = sizeA*sizeof(float);

	h_A = (float*)malloc(size);
	h_B = (float*)malloc(size);
	h_C = (float*)malloc(size);

	receive(h_A, f1, sizeA);
	receive(h_B, f2, sizeB);

	start = clock();
	sum(h_A, h_B, h_C, sizeA);
	end = clock();

	// print(h_C, sizeA);
	timeCPU = difftime(end, start);
	printf("CPU time: %.2lf\n", timeCPU);

	// --------------- GPU -----------------
	float *d_A, *d_B, *d_C, *h_C2;
	h_C2 = (float*)malloc(size);

	error = cudaMalloc((void**)&d_A, size);
	if (error != cudaSuccess) {
  	printf("Error allocating memory for d_A\n");
    exit(-1);
  }

	error = cudaMalloc((void**)&d_B, size);
	if (error != cudaSuccess) {
    printf("Error allocating memory for d_B\n");
  	exit(-1);
  }

	error = cudaMalloc((void**)&d_C, size);
	if (error != cudaSuccess) {
    printf("Error allocating memory for d_C\n");
	  exit(-1);
  }

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	double blockSize = 32.0;
	dim3 dimBlock(blockSize,1,1);
	dim3 dimGrid(ceil(sizeA / blockSize), 1, 1);

	start = clock();
	sumCU<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, sizeA);
	cudaDeviceSynchronize();
	end = clock();
	cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost);

	// print(h_C2, sizeA);
	timeGPU = difftime(end, start);
	printf("GPU time: %.2lf\n", timeGPU);

	if (compare(h_C, h_C2, sizeA)) {
		printf("Acceleration time: %f\n", timeCPU / timeGPU);
	} else printf("Error\n");

	free(h_A); free(h_B); free(h_C); free(h_C2);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	return 0;
}
 allocating memory for d_A\n");
    exit(-1);
  }

	error = cudaMalloc((void**)&d_B, size);
	if (error != cudaSuccess) {
    printf("Error allocating memory for d_B\n");
  	exit