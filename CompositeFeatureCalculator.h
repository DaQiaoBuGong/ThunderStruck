/**
 * thunderstruck/tracker: CompositeFeatureCalculator.h
 */

#ifndef H_THUNDERSTRUCK_COMPOSITEFEATURECALCULATOR
#define H_THUNDERSTRUCK_COMPOSITEFEATURECALCULATOR

#include <vector>

#include "FeatureCalculator.h"

namespace thunderstruck {

/**
 * \brief An instance of this class can be used to calculate multiple types of feature
 *        for rectangular samples from a frame in the tracking sequence and combine
 *        them into a larger feature vector.
 */
class CompositeFeatureCalculator : public FeatureCalculator
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The sub-calculators in the composite. */
  std::vector<FeatureCalculator_CPtr> m_calculators;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Adds a new calculator to the composite.
   *
   * The features calculated by this new calculator will follow those
   * of previously-added calculators in the feature vector.
   *
   * \param[in] calculator  The calculator to add.
   */
  void add_calculator(const FeatureCalculator_CPtr& calculator);

  /** Override */
  virtual void calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                  const cv::Rect_<float>& boundingBox, double *features, size_t offset, size_t stride,
                                  cudaTextureObject_t integralFrame, int threadsPerBlock) const;

  /** Override */
  virtual size_t feature_count() const;

  /** Override */
  virtual bool needs_integral_frame() const;
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<CompositeFeatureCalculator> CompositeFeatureCalculator_Ptr;

}

#endif
