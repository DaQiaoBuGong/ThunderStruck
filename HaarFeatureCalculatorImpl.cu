/**
 * thunderstruck/tracker: HaarFeatureCalculatorImpl.cu
 */

#include "HaarFeatureCalculatorImpl.h"
#include "cudaApi.cuh"

namespace thunderstruck {

//#################### CUDA KERNELS ####################

__global__ void ck_calculate_haar_features(cudaTextureObject_t integralFrame,
                                           float *sampleData, int *keepSamples, size_t sampleCount,
                                           float bbX, float bbY, float bbWidth, float bbHeight,
                                           float *bottoms, float *lefts, float *rights, float *tops, float *weights,
                                           double *features, size_t offset, size_t stride)
{
  int sampleIndex = blockIdx.x;
  int featureIndex = threadIdx.x;

  float bbArea = bbWidth * bbHeight;

  double result = 0.0;

  // 1 global read
  if(keepSamples[sampleIndex])
  {
    // Compute the coordinates of the sample.
    // 2 global reads
    float sampleLeft = bbX + sampleData[sampleIndex * 2];
    float sampleTop = bbY + sampleData[sampleIndex * 2 + 1];

    // Add up the weighted contributions from all the potential mini-boxes.
    for(int i = 0; i < 4; ++i)
    {
      // Compute the coordinates of the mini-box.
      int j = i * HAAR_FEATURE_COUNT + featureIndex;

      // 4 global reads
      float leftsJ = lefts[j];
      float topsJ = tops[j];
      int left(sampleLeft + leftsJ * bbWidth + 0.5f);
      int top(sampleTop + tops[j] * bbHeight + 0.5f);
      int right(left + (rights[j] - leftsJ) * bbWidth);
      int bottom(top + (bottoms[j] - topsJ) * bbHeight);

      // Add the weighted sum of the pixels in the mini-box to the result.
      // 1 global read, 4 texture reads
      result += weights[j] * (
          tex2D<int>(integralFrame, left, top)
        + tex2D<int>(integralFrame, right, bottom)
        - tex2D<int>(integralFrame, left, bottom)
        - tex2D<int>(integralFrame, right, top)
      );
    }

    // Normalize the result (divide it by the area of the sample).
    result /= bbArea;
  }

  // Determine the target position in the output array.
  size_t k = sampleIndex * stride + offset + featureIndex;

  // Write the result to the output array.
  features[k] = result;
}

//#################### WRAPPER FUNCTIONS ####################

void calculate_haar_features(cudaTextureObject_t integralFrame,
                             float *sampleData, int *keepSamples, size_t sampleCount,
                             float bbX, float bbY, float bbWidth, float bbHeight,
                             float *bottoms, float *lefts, float *rights, float *tops, float *weights,
                             double *features, size_t offset, size_t stride)
{
  ck_calculate_haar_features<<<sampleCount,HAAR_FEATURE_COUNT>>>(
    integralFrame, sampleData, keepSamples, sampleCount,
    bbX, bbY, bbWidth, bbHeight, bottoms, lefts, rights, tops, weights,
    features, offset, stride
  );
}

}
