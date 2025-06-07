#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 1024

__global__ void vecAdd(int* _a, int* _b, int* _c) {
    int tID = threadIdx.x;
    _c[tID] = _a[tID] + _b[tID];
}

int main(void) {
    int* d_a, * d_b, * d_c;
    int* a, * b, * c;
    
    int memSize = sizeof(int) * NUM_DATA;
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);

    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // memory alocation on the device
    cudaMalloc((void**)&d_a, memSize); cudaMemset(d_a, 0, memSize);
    cudaMalloc((void**)&d_b, memSize); cudaMemset(d_b, 0, memSize);
    cudaMalloc((void**)&d_c, memSize); cudaMemset(d_c, 0, memSize);

    // data copy: host -> device
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    // kernel launch
    vecAdd<<<1, NUM_DATA>>>(d_a, d_b, d_c);
    
    // data copy: device -> host
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    
    // release device memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    bool isCorrect = true;
    for (int i = 0; i < NUM_DATA; i++) {
        if (c[i] != a[i] + b[i]) {
            printf("Error at index %d\n", i);
            isCorrect = false;
        }
    }
    delete[] a; delete[] b; delete[] c;
    if (isCorrect) {
        printf("Success!\n");
    }
    return 0;
}