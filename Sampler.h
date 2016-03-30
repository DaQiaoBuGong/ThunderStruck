/**
 * thunderstruck/tracker: Sampler.h
 */

#ifndef H_THUNDERSTRUCK_SAMPLER
#define H_THUNDERSTRUCK_SAMPLER

#include <opencv2/opencv.hpp>

#include "GPUVector.h"
#define M_PI       3.14159265358979323846
namespace thunderstruck {

/**
 * \brief This struct contains helper functions for creating sample points and transferring them across to the GPU.
 */
struct Sampler
{
  //#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

  /**
   * \brief Transfers the specified sample points across to the GPU.
   *
   * \param[in] samples The sample points to transfer.
   * \return            A GPU vector containing a linearised version of the sample points.
   */
  static GPUVector<float> cpu_to_gpu(const std::vector<cv::Vec2f>& samples);
  
  /**
   * \brief Makes sample points at each pixel within a fixed radius of the origin.
   *
   * This is identical to the pixel-based sampling method used in the original version of Struck.
   *
   * \param[in] radius  The radius of the circle around the origin within which to sample.
   * \return            The sample points.
   */
  static std::vector<cv::Vec2f> make_pixel_samples(int radius);
  
  /**
   * \brief Makes sample points on a radial grid of fixed radius around the origin.
   *
   * This is identical to the radial sampling method used in the original version of Struck.
   *
   * \param[in] radius          The radius of the circle around the origin within which to sample.
   * \param[in] radialSegments  The number of radial segments into which to divide the circle.
   * \param[in] angularSegments The number of angular segments into which to divide the circle.
   * \return                    The sample points.
   */
  static std::vector<cv::Vec2f> make_radial_samples(float radius, int radialSegments, int angularSegments);
};

}

#endif
