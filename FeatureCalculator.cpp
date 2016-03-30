/**
 * thunderstruck/tracker: FeatureCalculator.cpp
 */
#include "stdafx.h"
#include "FeatureCalculator.h"

namespace thunderstruck {

//#################### DESTRUCTOR ####################

FeatureCalculator::~FeatureCalculator() {}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void FeatureCalculator::calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                           const cv::Rect_<float>& boundingBox, double *features, cudaTextureObject_t integralFrame,
                                           int threadsPerBlock) const
{
  calculate_features(
    frame, sampleData, keepSamples, sampleCount,
    boundingBox, features, 0, feature_count(),
    integralFrame, threadsPerBlock
  );
}

bool FeatureCalculator::needs_integral_frame() const
{
  return false;
}

}
