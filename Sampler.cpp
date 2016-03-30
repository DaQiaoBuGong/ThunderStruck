/**
 * thunderstruck/tracker: Sampler.cpp
 */
#include "stdafx.h"
#include "Sampler.h"

namespace thunderstruck {

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

GPUVector<float> Sampler::cpu_to_gpu(const std::vector<cv::Vec2f>& samples)
{
  // Linearise the sample points into a float vector on the CPU.
  size_t sampleCount = samples.size();
  std::vector<float> cpuVec(sampleCount * 2);
  size_t k = 0;
  for(size_t i = 0; i < sampleCount; ++i)
  {
    cpuVec[k++] = samples[i][0];
    cpuVec[k++] = samples[i][1];
  }
  
  // Transfer the float vector across to the GPU.
  return make_gpu_vector(cpuVec);
}

std::vector<cv::Vec2f> Sampler::make_pixel_samples(int radius)
{
  std::vector<cv::Vec2f> result;
  
  result.push_back(cv::Vec2f(0.0f, 0.0f));
  
  int radiusSquared = radius * radius;
  for(int y = -radius; y <= radius; ++y)
  {
    for(int x = -radius; x <= radius; ++x)
    {
      if(x*x + y*y > radiusSquared) continue;
      if(x == 0 && y == 0) continue;
      
      result.push_back(cv::Vec2f(x, y));
    }
  }
  
  return result;
}

std::vector<cv::Vec2f> Sampler::make_radial_samples(float radius, int radialSegments, int angularSegments)
{
  std::vector<cv::Vec2f> result;
  
  result.push_back(cv::Vec2f(0.0f, 0.0f));
  
  float radialStep = radius / radialSegments;
  float angularStep = static_cast<float>(2.0 * M_PI / angularSegments);
  for(int radialIndex = 1; radialIndex <= radialSegments; ++radialIndex)
  {
    float phase = (radialIndex % 2) * angularStep / 2.0f;
    for(int angularIndex = 0; angularIndex < angularSegments; ++angularIndex)
    {
      float scale = radialIndex * radialStep;
      float angle = angularIndex * angularStep + phase;
      result.push_back(cv::Vec2f(scale * cosf(angle), scale * sinf(angle)));
    }
  }
  
  return result;
}

}
