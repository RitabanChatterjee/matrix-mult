#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

#ifndef TILE
#define TILE 16
#endif

__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 512;
    int repeats = argc > 2 ? atoi(argv[2]) : 20;
    if (N <= 0 || repeats <= 0) { fprintf(stderr, "Usage: %s [N>0] [repeats>0]\n", argv[0]); return 1; }
    size_t bytes = (size_t)N * N * sizeof(float);
    float *A = (float*)malloc(bytes), *B = (float*)malloc(bytes), *C = (float*)malloc(bytes);
    if (!A || !B || !C) { fprintf(stderr,"Host allocation failed\n"); return 1; }
    for (size_t i=0; i<(size_t)N*N; ++i) { A[i]=1.0f; B[i]=1.0f; }
    float *dA, *dB, *dC;
    CHECK(cudaMalloc((void**)&dA, bytes)); CHECK(cudaMalloc((void**)&dB, bytes)); CHECK(cudaMalloc((void**)&dC, bytes));
    CHECK(cudaMemcpy(dA,A,bytes,cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(dB,B,bytes,cudaMemcpyHostToDevice));
    dim3 block(TILE,TILE), grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);
    for(int i=0;i<5;++i) matmulTiled<<<grid,block>>>(dA,dB,dC,N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for(int i=0;i<repeats;++i) matmulTiled<<<grid,block>>>(dA,dB,dC,N);
    CHECK(cudaGetLastError()); CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
    float ms; CHECK(cudaEventElapsedTime(&ms,start,stop));
    CHECK(cudaMemcpy(C,dC,bytes,cudaMemcpyDeviceToHost));
    size_t errors=0; for(size_t i=0;i<(size_t)N*N;++i) if(fabsf(C[i]-(float)N)>1e-3f) ++errors;
    double avgMs=ms/repeats, gflops=2.0*(double)N*N*N/(avgMs*1e6);
    printf("N=%d TILE=%d repeats=%d\n",N,TILE,repeats);
    printf("Verification: %s (C[0]=%.1f, expected=%d, mismatches=%zu)\n",errors?"FAILED":"PASSED",C[0],N,errors);
    printf("Average kernel time: %.4f ms\nPerformance: %.2f GFLOP/s\n",avgMs,gflops);
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC)); free(A); free(B); free(C);
    return errors ? 1 : 0;
}
