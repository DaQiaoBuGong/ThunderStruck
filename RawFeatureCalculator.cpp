/**
 * thunderstruck/tracker: RawFeatureCalculator.cpp
 */
#include "stdafx.h"
#include "RawFeatureCalculator.h"

#include "RawFeatureCalculatorImpl.h"

namespace thunderstruck {

//#################### PUBLIC MEMBER FUNCTIONS ####################

void RawFeatureCalculator::calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                              const cv::Rect_<float>& boundingBox, double *features, size_t offset, size_t stride,
                                              cudaTextureObject_t integralFrame, int threadsPerBlock) const
{
  calculate_raw_features(
    frame, sampleData, keepSamples, sampleCount,
    boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height,
    features, offset, stride,
    threadsPerBlock
  );
}

size_t RawFeatureCalculator::feature_count() const
{
  return RAW_FEATURE_COUNT;
}

}
