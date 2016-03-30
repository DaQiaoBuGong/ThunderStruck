/**
 * thunderstruck/tracker: FeatureCalculator.h
 */

#ifndef H_THUNDERSTRUCK_FEATURECALCULATOR
#define H_THUNDERSTRUCK_FEATURECALCULATOR

#include <boost/shared_ptr.hpp>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

namespace thunderstruck {

/**
 * \brief An instance of a class deriving from this one can be used to calculate features
 *        for rectangular samples from a frame in the tracking sequence.
 */
class FeatureCalculator
{
  //#################### DESTRUCTOR ####################
public:
  /**
   * \brief Destroys the feature calculator.
   */
  virtual ~FeatureCalculator();

  //#################### PUBLIC ABSTRACT MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Calculates features for each of a set of samples that is within the bounds of the specified frame.
   *
   * \param[in]  frame            The texture object representing the frame on the GPU.
   * \param[in]  sampleData       A linearised version of the sample points (expressed relative to a central bounding box).
   * \param[in]  keepSamples      An array of flags specifying whether or not each sample is within the bounds of the frame.
   * \param[in]  sampleCount      The total number of samples (including those that fall outside the bounds of the frame).
   * \param[in]  boundingBox      The central bounding box to be used as the anchor for the samples.
   * \param[out] features         The array into which the calculated features should be written.
   * \param[in]  offset           The offset from the start of the features array at which to start writing.
   * \param[in]  stride           The stride in the features array between one sample and the next.
   * \param[in]  integralFrame    The texture object representing the integral image for the frame (if needed) on the GPU.
   * \param[in]  threadsPerBlock  The number of threads to use in each CUDA block.
   */
  virtual void calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                  const cv::Rect_<float>& boundingBox, double *features, size_t offset, size_t stride,
                                  cudaTextureObject_t integralFrame, int threadsPerBlock) const = 0;

  /**
   * \brief Gets the number of features calculated by this feature calculator.
   *
   * \return  The number of features calculated by this feature calculator.
   */
  virtual size_t feature_count() const = 0;

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Calculates features for each of a set of samples that is within the bounds of the specified frame.
   *
   * \param[in]  frame            The texture object representing the frame on the GPU.
   * \param[in]  sampleData       A linearised version of the sample points (expressed relative to a central bounding box).
   * \param[in]  keepSamples      An array of flags specifying whether or not each sample is within the bounds of the frame.
   * \param[in]  sampleCount      The total number of samples (including those that fall outside the bounds of the frame).
   * \param[in]  boundingBox      The central bounding box to be used as the anchor for the samples.
   * \param[out] features         The array into which the calculated features should be written.
   * \param[in]  integralFrame    The texture object representing the integral image for the frame (if needed) on the GPU.
   * \param[in]  threadsPerBlock  The number of threads to use in each CUDA block.
   */
  void calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                          const cv::Rect_<float>& boundingBox, double *features, cudaTextureObject_t integralFrame,
                          int threadsPerBlock = 128) const;

  /**
   * \brief Gets whether or not this feature calculator needs an integral image for each frame.
   *
   * \return  true, if this feature calculator needs an integral image, or false otherwise.
   */
  virtual bool needs_integral_frame() const;
};

//#################### TYPEDEFS ####################

typedef boost::shared_ptr<FeatureCalculator> FeatureCalculator_Ptr;
typedef boost::shared_ptr<const FeatureCalculator> FeatureCalculator_CPtr;

}

#endif
