#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"

#define NUM_DATA 1024*1024

__global__ void vecAdd(int* _a, int* _b, int* _c, int n) {
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < n) {
        _c[tID] = _a[tID] + _b[tID];
    }
}

int main(void) {
    // set timer
    DS_timer timer(5);
    timer.setTimerName(0, (char *)"CUDA Total");
    timer.setTimerName(1, (char *)"Computation(Kernel)");
    timer.setTimerName(2, (char *)"Data Trans. : Host -> Device");
    timer.setTimerName(3, (char *)"Data Trans. : Device -> Host");
    timer.setTimerName(4, (char *)"VecAdd on Host");
    timer.initTimers();

    int* d_a, * d_b, * d_c;
    int* a, * b, * c, * hc;
    
    int memSize = sizeof(int) * NUM_DATA;
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    hc = new int[NUM_DATA]; memset(hc, 0, memSize);


    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // vector sum on host
    timer.onTimer(4);
    for (int i = 0; i < NUM_DATA; i++) {
        hc[i] = a[i] + b[i];
    }
    timer.offTimer(4);

    // memory alocation on the device
    cudaMalloc((void**)&d_a, memSize); cudaMemset(d_a, 0, memSize);
    cudaMalloc((void**)&d_b, memSize); cudaMemset(d_b, 0, memSize);
    cudaMalloc((void**)&d_c, memSize); cudaMemset(d_c, 0, memSize);

    timer.onTimer(0);

    // data copy: host -> device
    timer.onTimer(2);
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    // kernel launch
    timer.onTimer(1);
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_DATA + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, NUM_DATA);
    cudaDeviceSynchronize(); // Wait for the kernel to complete
    timer.offTimer(1);
    
    // data copy: device -> host
    timer.onTimer(3);
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    timer.offTimer(3);
    
    timer.offTimer(0);

    // release device memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    timer.printTimer();
    
    bool isCorrect = true;
    for (int i = 0; i < NUM_DATA; i++) {
        if (c[i] != hc[i]) {
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