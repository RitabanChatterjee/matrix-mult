#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>

#define CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)

struct ThreadStats {
    unsigned long long cycles;
    unsigned int multiplies;
    unsigned int additions;
    unsigned int logicalLoads;
    unsigned int logicalStores;
    int blockX, blockY, threadX, threadY;
};

__global__ void matmul(const float* A, const float* B, float* C, ThreadStats* stats, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    unsigned long long start = clock64();
    float sum = 0.0f;
    unsigned int muls=0, adds=0, loads=0;
    for (int k=0; k<N; ++k) {
        float a=A[row*N+k];
        float b=B[k*N+col];
        sum += a*b;
        ++muls; ++adds; loads+=2;
    }
    C[row*N+col]=sum;
    unsigned long long end = clock64();
    int id=row*N+col;
    stats[id]={end-start,muls,adds,loads,1,(int)blockIdx.x,(int)blockIdx.y,(int)threadIdx.x,(int)threadIdx.y};
}

int main(int argc,char** argv) {
    int N=argc>1?std::atoi(argv[1]):256;
    if(N<=0 || N>2048){fprintf(stderr,"Choose N between 1 and 2048\n");return 1;}
    size_t count=(size_t)N*N, bytes=count*sizeof(float);
    std::vector<float> A(count,1.0f),B(count,2.0f),C(count);
    std::vector<ThreadStats> stats(count);
    float *dA,*dB,*dC; ThreadStats* dStats;
    CHECK(cudaMalloc((void**)&dA,bytes)); CHECK(cudaMalloc((void**)&dB,bytes));
    CHECK(cudaMalloc((void**)&dC,bytes)); CHECK(cudaMalloc((void**)&dStats,count*sizeof(ThreadStats)));
    CHECK(cudaMemcpy(dA,A.data(),bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,B.data(),bytes,cudaMemcpyHostToDevice));
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    matmul<<<grid,block>>>(dA,dB,dC,dStats,N);
    CHECK(cudaGetLastError()); CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
    float ms=0; CHECK(cudaEventElapsedTime(&ms,start,stop));
    CHECK(cudaMemcpy(C.data(),dC,bytes,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(stats.data(),dStats,count*sizeof(ThreadStats),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<count;++i) if(C[i] != 2.0f*N) {fprintf(stderr,"Incorrect result at %zu: %f\n",i,C[i]);return 1;}
    std::ofstream out("thread_stats.csv");
    out<<"row,col,block_x,block_y,thread_x,thread_y,multiplications,additions,logical_loads,logical_stores,logical_bytes,elapsed_clock_cycles\n";
    unsigned long long minC=~0ULL,maxC=0; long double sumC=0;
    for(int row=0;row<N;++row) for(int col=0;col<N;++col) {
        const auto& s=stats[(size_t)row*N+col];
        minC=std::min(minC,s.cycles);maxC=std::max(maxC,s.cycles);sumC+=s.cycles;
        out<<row<<','<<col<<','<<s.blockX<<','<<s.blockY<<','<<s.threadX<<','<<s.threadY<<','<<s.multiplies<<','<<s.additions<<','<<s.logicalLoads<<','<<s.logicalStores<<','<<4ULL*(s.logicalLoads+s.logicalStores)<<','<<s.cycles<<'\n';
    }
    out.close();
    std::ofstream summary("summary.txt");
    summary<<std::fixed<<std::setprecision(4);
    summary<<"Matrix dimension: "<<N<<" x "<<N<<"\nOutput elements / active threads: "<<count<<"\nBlock dimensions: 16 x 16\nGrid dimensions: "<<grid.x<<" x "<<grid.y<<"\nKernel elapsed time (CUDA events): "<<ms<<" ms\nMinimum per-thread elapsed clock cycles: "<<minC<<"\nMaximum per-thread elapsed clock cycles: "<<maxC<<"\nMean per-thread elapsed clock cycles: "<<(double)(sumC/count)<<"\nTotal logical floating point operations: "<<2ULL*N*N*N<<"\nEstimated arithmetic throughput: "<<(2.0*N*N*N)/(ms*1e6)<<" GFLOP/s\nNote: logical memory bytes are source-level requests, not DRAM traffic or cache misses.\nNote: per-thread clock64 differences include stalls/scheduling and cannot be summed to obtain kernel time.\nNote: multiply-add may be compiled to FMA; operations are algorithmic counts.\n";
    summary.close();
    printf("Verified C[i] = %g; kernel time = %.4f ms\nWrote thread_stats.csv and summary.txt\n",C[0],ms);
    CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dA));CHECK(cudaFree(dB));CHECK(cudaFree(dC));CHECK(cudaFree(dStats));
}
