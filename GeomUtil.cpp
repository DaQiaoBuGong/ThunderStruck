/**
 * thunderstruck/util: GeomUtil.cpp
 */
#include "stdafx.h"
#include "GeomUtil.h"

namespace thunderstruck {

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

double GeomUtil::compute_overlap(const cv::Rect_<float>& bb1, const cv::Rect_<float>& bb2)
{
  float x0 = std::max(bb1.x, bb2.x);
  float x1 = std::min(bb1.x + bb1.width, bb2.x + bb2.width);
  float y0 = std::max(bb1.y, bb2.y);
  float y1 = std::min(bb1.y + bb1.height, bb2.y + bb2.height);
  if(x0 >= x1 || y0 >= y1) return 0.0f;
  float areaInt = (x1 - x0) * (y1 - y0);
  return areaInt / (bb1.area() + bb2.area() - areaInt);
}

}
