/**
 * thunderstruck/util: GeomUtil.h
 */

#ifndef H_THUNDERSTRUCK_GEOMUTIL
#define H_THUNDERSTRUCK_GEOMUTIL

#include <opencv2/opencv.hpp>

namespace thunderstruck {

/**
 * \brief This struct contains geometric utility functions.
 */
struct GeomUtil
{
  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################
public:
  /**
   * \brief Computes the amount of overlap between two bounding boxes.
   *
   * The result is a number between 0 and 1, with 0 meaning that the boxes
   * do not overlap at all, and 1 meaning that they completely overlap.
   *
   * \param[in] bb1 The first bounding box.
   * \param[in] bb2 The second bounding box.
   * \return        The amount of overlap between the two bounding boxes.
   */
  static double compute_overlap(const cv::Rect_<float>& bb1, const cv::Rect_<float>& bb2);
};

}

#endif
