/**
 * thunderstruck/tracker: RawFeatureCalculatorImpl.cu
 */

#include "RawFeatureCalculatorImpl.h"
#include "cudaApi.cuh"

namespace thunderstruck {

//#################### CUDA KERNELS ####################

__global__ void ck_calculate_raw_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                          float bbX, float bbY, float bbWidth, float bbHeight,
                                          double *features, size_t offset, size_t stride,
                                          size_t numThreads)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if(tid < numThreads)
  {
    // Determine which sample we're processing and the feature index within that sample.
    size_t sampleIndex = tid / RAW_FEATURE_COUNT;
    size_t featureIndex = tid % RAW_FEATURE_COUNT;

    // Determine the source position in the frame.
    size_t yIndex = featureIndex / PATCH_SIZE;
    size_t xIndex = featureIndex % PATCH_SIZE;
    int x = int(bbX + sampleData[sampleIndex * 2]) + int(bbWidth * xIndex / (PATCH_SIZE - 1));
    int y = int(bbY + sampleData[sampleIndex * 2 + 1]) + int(bbHeight * yIndex / (PATCH_SIZE - 1));

    // Determine the target position in the output array.
    size_t k = sampleIndex * stride + offset + featureIndex;

    // Write a scaled version of the source value into the output array at the target position.
    features[k] = keepSamples[sampleIndex] ? tex2D<unsigned char>(frame, x, y) / 255.0 : 0.0;
  }
}

//#################### WRAPPER FUNCTIONS ####################

void calculate_raw_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                            float bbX, float bbY, float bbWidth, float bbHeight,
                            double *features, size_t offset, size_t stride, int threadsPerBlock)
{
  int numThreads = sampleCount * RAW_FEATURE_COUNT;
  int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
  ck_calculate_raw_features<<<numBlocks,threadsPerBlock>>>(
    frame, sampleData, keepSamples, sampleCount,
    bbX, bbY, bbWidth, bbHeight,
    features, offset, stride,
    numThreads
  );
}

}
