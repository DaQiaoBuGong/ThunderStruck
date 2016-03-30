/**
 * thunderstruck/tracker: Tracker.h
 */

#ifndef H_THUNDERSTRUCK_TRACKER
#define H_THUNDERSTRUCK_TRACKER

#include <boost/shared_ptr.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <opencv2/opencv.hpp>

#include "SVM.h"
#include "FeatureCalculator.h"
#include "GPUTexture1.h"
#include "GPUVector.h"

namespace thunderstruck {

/**
 * \brief An instance of this class can be used to track an object through a sequence of frames.
 */
class Tracker
{
  //#################### PRIVATE VARIABLES ####################
private:
  /** The current bounding box around the object. */
  cv::Rect_<float> m_curBoundingBox;

  /** The feature calculator to use. */
  FeatureCalculator_CPtr m_featureCalculator;

  /** The SVM on which the tracker is based. */
  SVM m_svm;

  /** The number of sample points for tracking. */
  size_t m_trackingSampleCount;

  /** A GPU vector containing a linearised version of the sample points for tracking. */
  GPUVector<float> m_trackingSampleData;

  //#################### CONSTRUCTORS ####################
public:
  /**
   * \brief Constructs a tracker.
   *
   * \param[in] initialBoundingBox  The initial bounding box around the object.
   * \param[in] featureCalculator   The feature calculator to use.
   */
  Tracker(const cv::Rect_<float>& initialBoundingBox, const FeatureCalculator_CPtr& featureCalculator);

  //#################### PUBLIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Gets the current bounding box around the object.
   *
   * \return  The current bounding box around the object.
   */
  const cv::Rect_<float>& get_current_bounding_box() const;

  /**
   * \brief Updates the tracker with the current frame in the sequence.
   *
   * \param[in] frame         The frame with which to update the tracker.
   * \param[in] initialFrame  Whether or not the current frame is the initial one.
   */
  void update(const cv::Mat& frame, bool initialFrame);

  //#################### PRIVATE MEMBER FUNCTIONS ####################
private:
  /**
   * \brief Attempts to predict a bounding box around the object being tracked in the next frame
   *        in the sequence. If this is successful, the tracker's current bounding box is updated
   *        accordingly.
   *
   * \param[in] frame         The next frame in the sequence.
   * \param[in] integralFrame The integral image for the frame, if needed by the feature calculator.
   * \return                  true, if a bounding box was successfully predicted for the next frame,
   *                          or false otherwise.
   */
  bool predict_bounding_box(const GPUTexture1b& frame, const boost::optional<GPUTexture1i>& integralFrame);
};

}

#endif
