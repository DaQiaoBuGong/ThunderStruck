/**
 * thunderstruck/tracker: RawFeatureCalculator.h
 */

#ifndef H_THUNDERSTRUCK_RAWFEATURECALCULATOR
#define H_THUNDERSTRUCK_RAWFEATURECALCULATOR

#include "FeatureCalculator.h"

namespace thunderstruck {

/**
 * \brief An instance of this class can be used to calculate raw features
 *        for rectangular samples from a frame in the tracking sequence.
 */
class RawFeatureCalculator : public FeatureCalculator
{
  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                  const cv::Rect_<float>& boundingBox, double *features, size_t offset, size_t stride,
                                  cudaTextureObject_t integralFrame, int threadsPerBlock) const;

  /** Override */
  virtual size_t feature_count() const;
};

}

#endif
