/**
 * thunderstruck/tracker: CompositeFeatureCalculator.cpp
 */
#include "stdafx.h"
#include "CompositeFeatureCalculator.h"

#include <cassert>

namespace thunderstruck {

//#################### PUBLIC MEMBER FUNCTIONS ####################

void CompositeFeatureCalculator::add_calculator(const FeatureCalculator_CPtr& calculator)
{
  assert(calculator.get() != NULL);
  m_calculators.push_back(calculator);
}

void CompositeFeatureCalculator::calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                                    const cv::Rect_<float>& boundingBox, double *features, size_t offset, size_t stride,
                                                    cudaTextureObject_t integralFrame, int threadsPerBlock) const
{
  size_t cumulativeFeatureCount = 0;
  for(std::vector<FeatureCalculator_CPtr>::const_iterator it = m_calculators.begin(), iend = m_calculators.end(); it != iend; ++it)
  {
    (*it)->calculate_features(frame, sampleData, keepSamples, sampleCount, boundingBox, features, offset + cumulativeFeatureCount, stride, integralFrame, threadsPerBlock);
    cumulativeFeatureCount += (*it)->feature_count();
  }
}

size_t CompositeFeatureCalculator::feature_count() const
{
  size_t result = 0;
  for(std::vector<FeatureCalculator_CPtr>::const_iterator it = m_calculators.begin(), iend = m_calculators.end(); it != iend; ++it)
  {
    result += (*it)->feature_count();
  }
  return result;
}

bool CompositeFeatureCalculator::needs_integral_frame() const
{
  for(std::vector<FeatureCalculator_CPtr>::const_iterator it = m_calculators.begin(), iend = m_calculators.end(); it != iend; ++it)
  {
    if((*it)->needs_integral_frame()) return true;
  }
  return false;
}

}
