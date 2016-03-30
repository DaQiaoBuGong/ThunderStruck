/**
 * thunderstruck/tracker: SampleFilterer.cu
 */

#include "SampleFilterer.h"
#include "cudaApi.cuh"

namespace thunderstruck {

//#################### CUDA_KERNELS ####################

__global__ void ck_filter_samples(float *sampleData, size_t sampleCount, int *keepSamples,
                                  float bbX, float bbY, float bbWidth, float bbHeight,
                                  int frameWidth, int frameHeight)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < sampleCount)
  {
    float offsetX = sampleData[tid * 2]; // the sample data array has the form [x1,y1,x2,y2,...]
    float offsetY = sampleData[tid * 2 + 1];
    int minX(bbX + offsetX);
    int maxX(minX + bbWidth);
    int minY(bbY + offsetY);
    int maxY(minY + bbHeight);
    keepSamples[tid] = minX >= 0 && maxX < frameWidth && minY >= 0 && maxY < frameHeight ? 1 : 0;
  }
}

//#################### WRAPPER FUNCTIONS ####################

void filter_samples(float *sampleData, size_t sampleCount, int *keepSamples,
                    float bbX, float bbY, float bbWidth, float bbHeight,
                    int frameWidth, int frameHeight, int threadsPerBlock)
{
  int numBlocks = (sampleCount + threadsPerBlock - 1) / threadsPerBlock;
  ck_filter_samples<<<numBlocks,threadsPerBlock>>>(sampleData, sampleCount, keepSamples, bbX, bbY, bbWidth, bbHeight, frameWidth, frameHeight);
}

}
