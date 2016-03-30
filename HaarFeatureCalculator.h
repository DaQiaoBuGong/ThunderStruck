/**
 * thunderstruck/tracker: HaarFeatureCalculator.h
 */

#ifndef H_THUNDERSTRUCK_HAARFEATURECALCULATOR
#define H_THUNDERSTRUCK_HAARFEATURECALCULATOR

#include "FeatureCalculator.h"
#include "GPUVector.h"

namespace thunderstruck {

/**
 * \brief An instance of this class can be used to calculate Haar features
 *        for rectangular samples from a frame in the tracking sequence.
 */
class HaarFeatureCalculator : public FeatureCalculator
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** TODO */
  std::vector<float> m_bottomsCPU;

  /** TODO */
  GPUVector<float> m_bottomsGPU;

  /** TODO */
  std::vector<float> m_leftsCPU;

  /** TODO */
  GPUVector<float> m_leftsGPU;

  /** TODO */
  std::vector<float> m_rightsCPU;

  /** TODO */
  GPUVector<float> m_rightsGPU;

  /** TODO */
  std::vector<float> m_topsCPU;

  /** TODO */
  GPUVector<float> m_topsGPU;

  /** TODO */
  std::vector<float> m_weightsCPU;

  /** TODO */
  GPUVector<float> m_weightsGPU;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs an instance of the feature calculator.
   */
  HaarFeatureCalculator();

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /** Override */
  virtual void calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                  const cv::Rect_<float>& boundingBox, double *features, size_t offset, size_t stride,
                                  cudaTextureObject_t integralFrame, int threadsPerBlock) const;

  /** Override */
  virtual size_t feature_count() const;

  /** Override */
  virtual bool needs_integral_frame() const;

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /** TODO */
  void set_values(int index, const cv::Rect_<float>& minibox, float weight, int type, float boxArea);
};

}

#endif
