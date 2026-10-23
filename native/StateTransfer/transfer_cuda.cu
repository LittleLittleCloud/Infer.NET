// Transport microbenchmark only: synthetic GPU-resident state, no model execution.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
static unsigned char *deviceData, *staging;
static unsigned int *errors;
static size_t capacity, chunkCapacity;
static char errorMessage[256];
#define API extern "C" __declspec(dllexport)
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){snprintf(errorMessage,sizeof(errorMessage),"%s: %s",#x,cudaGetErrorString(e));return -1;} } while(0)
__device__ uint32_t pattern(uint32_t i,uint32_t seed){return (i*1664525u+1013904223u)^seed;}
__global__ void fill(uint32_t *p,size_t count,uint32_t seed){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)p[i]=pattern(uint32_t(i),seed);}
__global__ void verify(const uint32_t *p,size_t count,uint32_t seed,unsigned int *bad){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count && p[i]!=pattern(uint32_t(i),seed))atomicAdd(bad,1u);}
API const char *st_error(){return errorMessage;}
API int st_open(size_t bytes,size_t chunk){
 if(!bytes || bytes%4 || !chunk || chunk%4)return -1;
 capacity=bytes;chunkCapacity=chunk;
 CHECK(cudaMalloc(&deviceData,bytes));CHECK(cudaMallocHost(&staging,chunk));CHECK(cudaMalloc(&errors,4));return 0;
}
API int st_prepare(size_t bytes,uint32_t seed,int source){
 if(bytes>capacity || bytes%4)return -1;
 if(source)fill<<<(bytes/4+255)/256,256>>>((uint32_t*)deviceData,bytes/4,seed);
 else CHECK(cudaMemset(deviceData,0,bytes));
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());return 0;
}
API void *st_pointer(size_t offset){return staging;}
API int st_read(size_t offset,size_t bytes){if(offset>capacity || bytes>capacity-offset || bytes>chunkCapacity)return -1;CHECK(cudaMemcpy(staging,deviceData+offset,bytes,cudaMemcpyDeviceToHost));return 0;}
API int st_write(size_t offset,size_t bytes){if(offset>capacity || bytes>capacity-offset || bytes>chunkCapacity)return -1;CHECK(cudaMemcpy(deviceData+offset,staging,bytes,cudaMemcpyHostToDevice));return 0;}
API int st_validate(size_t bytes,uint32_t seed){
 if(bytes>capacity || bytes%4)return -1;
 CHECK(cudaMemset(errors,0,4));verify<<<(bytes/4+255)/256,256>>>((uint32_t*)deviceData,bytes/4,seed,errors);
 CHECK(cudaGetLastError());unsigned int bad=0;CHECK(cudaMemcpy(&bad,errors,4,cudaMemcpyDeviceToHost));
 if(bad){snprintf(errorMessage,sizeof(errorMessage),"%u mismatched uint32 words",bad);return -1;}return 0;
}
API void st_close(){cudaFree(errors);cudaFreeHost(staging);cudaFree(deviceData);}
