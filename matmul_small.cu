#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DO_CPU
#define DATA_TYPE float

// Matrix size
#define SIZE_M (32)
#define SIZE_N (32)
#define SIZE_K (128)

#define BLOCK_SIZE 32
// int - 32: 55.81ms, 16: 17.47ms, 8: 10.43ms	
// float - 32: 56.48ms, 16: 17.78ms, 8: 29.55ms

template<class T> void allocNinitMem(T** p, long long size, double* memUsage = NULL);
bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size);

/******************************************************************
* Complete this kernels
******************************************************************/
__global__ void MatMulSmallShared(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	// Write your kernel here
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;
	__shared__ float sA[SIZE_M][SIZE_K];
	__shared__ float sB[SIZE_K][SIZE_N];

	if (row == 0) {
		for (int i = 0; i < k; i++)
			sB[i][col] = matB[i * n + col];
	}
	if (col == 0) {
		for (int i = 0; i < k; i++)
			sA[row][i] = matA[row * k + i];
	}
	__syncthreads();
	
	float result = 0;
	for (int i = 0; i < k; i++)
		result += __fmul_rn(sA[row][i], sB[i][col]);
	matC[index] = result;
}


int main(int argc, char* argv[])
{
	DS_timer timer(10);
	timer.setTimerName(0, (char*)"CPU code");
	timer.setTimerName(1, (char*)"Kernel");
	timer.setTimerName(2, (char*)"[Data transter] host->device");
	timer.setTimerName(3, (char*)"[Data transfer] device->host");
	timer.setTimerName(4, (char*)"GPU total");

	// set matrix size
	int m, n, k;
	m = SIZE_M;
	n = SIZE_N;
	k = SIZE_K;

	printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

	int sizeA = m * k;
	int sizeB = k * n;
	int sizeC = m * n;

	// Make matrix
	DATA_TYPE* A = NULL, * B = NULL;
	allocNinitMem<DATA_TYPE>(&A, sizeA);
	allocNinitMem<DATA_TYPE>(&B, sizeB);

	DATA_TYPE* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<DATA_TYPE>(&Ccpu, sizeC);
	allocNinitMem<DATA_TYPE>(&Cgpu, sizeC);

	// generate input matrices
	for (int i = 0; i < sizeA; i++) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	for (int i = 0; i < sizeB; i++) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

	// CPU algorithm
	timer.onTimer(0);
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int cIndex = row * n + col;
			Ccpu[cIndex] = 0;
			for (int i = 0; i < k; i++)
				Ccpu[cIndex] += (A[row * k + i] * B[i * n + col]);
		}
	}
	printf("CPU finished!\n");
	timer.offTimer(0);

	timer.onTimer(4);
	/******************************************************************
	* Write your codes for GPU algorithm from here
	******************************************************************/
	DATA_TYPE* dA, * dB, * dC;

	// 1. Allocate device memory for dA, dB, dC
	// Hint: cudaMalloc, cudaMemset
	cudaMalloc((void**)&dA, sizeA * sizeof(DATA_TYPE));
	cudaMemset(dA, 0, sizeA * sizeof(DATA_TYPE));
	cudaMalloc((void**)&dB, sizeB * sizeof(DATA_TYPE));
	cudaMemset(dB, 0, sizeB * sizeof(DATA_TYPE));
	cudaMalloc((void**)&dC, sizeC * sizeof(DATA_TYPE));
	cudaMemset(dC, 0, sizeC * sizeof(DATA_TYPE));


	timer.onTimer(2);

	// 2. Send(Copy) the input matrices to GPU (A -> dB, B -> dB)
	// Hint: cudaMemcpy
	cudaMemcpy(dA, A, sizeA * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(dC, Cgpu, sizeC * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	timer.offTimer(2);

	// 3. Set the thread layout
	// 
	// dim3 gridDim(?, ?);
	// dim3 blockDim(?, ?);

	dim3 gridDim(1);
	dim3 blockDim(32, 32);
	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	timer.onTimer(1);

	// 4. Kernel call
	MatMulSmallShared <<< gridDim, blockDim >>> (dA, dB, dC, m, n, k);

	cudaDeviceSynchronize(); // this is synchronization for mearusing the kernel processing time
	timer.offTimer(1);

	timer.onTimer(3);

	//5. Get(copy) the result from GPU to host memory (dC -> Cgpu)
	// Hint: cudaMemcpy
	cudaMemcpy(Cgpu, dC, sizeC * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

	timer.offTimer(3);

	// 6. Release device memory space (dA, dB, dC)
	// Hint: cudaFree
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	/******************************************************************
	******************************************************************/
	timer.offTimer(4);

	compareMatrix(Ccpu, Cgpu, sizeC);
	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}


// Utility functions
bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size)
{
	bool isMatched = true;
	for (int i = 0; i < _size; i++) {
		if (_A[i] != _B[i]) {
			printf("[%d] not matched! (%f, %f)\n", i, _A[i], _B[i]);
			getchar();
			isMatched = false;
		}
	}
	if (isMatched)
		printf("Results are matched!\n");
	else
		printf("Results are not matched!!!!!!!!!!!\n");

	return isMatched;
}

template<class T>
void allocNinitMem(T** p, long long size, double* memUsage) {
	*p = new T[size];
	memset(*p, 0, sizeof(T) * size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T) * size;
	}
}