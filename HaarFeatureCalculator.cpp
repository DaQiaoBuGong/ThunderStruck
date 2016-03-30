/**
 * thunderstruck/tracker: HaarFeatureCalculator.cpp
 */
#include "stdafx.h"
#include "HaarFeatureCalculator.h"

#include "HaarFeatureCalculatorImpl.h"

namespace thunderstruck {

//#################### CONSTRUCTORS ####################

HaarFeatureCalculator::HaarFeatureCalculator()
{
  // Generate the coordinates and weights for the mini-boxes that can be overlaid over each sample to calculate Haar features.
  float xs[] = { 0.2f, 0.4f, 0.6f, 0.8f };
  float ys[] = { 0.2f, 0.4f, 0.6f, 0.8f };
  float scales[] = { 0.2f, 0.4f };

  const size_t maxMiniBoxes = 4;
  const size_t size = HAAR_FEATURE_COUNT * maxMiniBoxes;
  m_bottomsCPU.resize(size);
  m_leftsCPU.resize(size);
  m_rightsCPU.resize(size);
  m_topsCPU.resize(size);
  m_weightsCPU.resize(size);

  int featureID = 0;
  for(int y = 0; y < 4; ++y)
  {
    for(int x = 0; x < 4; ++x)
    {
      for(int s = 0; s < 2; ++s)
      {
        cv::Rect_<float> box(xs[x] - scales[s]/2, ys[y] - scales[s]/2, scales[s], scales[s]);
        float boxArea = box.area();
        for(int type = 0; type < 6; ++type)
        {
          switch(type)
          {
            case 0:
              set_values(featureID + 0 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y, box.width, box.height/2), 1.0f, type, boxArea);
              set_values(featureID + 1 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y + box.height/2, box.width, box.height/2), -1.0f, type, boxArea);
              break;
            case 1:
              set_values(featureID + 0 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y, box.width/2, box.height), 1.0f, type, boxArea);
              set_values(featureID + 1 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x + box.width/2, box.y, box.width/2, box.height), -1.0f, type, boxArea);
              break;
            case 2:
              set_values(featureID + 0 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y, box.width/3, box.height), 1.0f, type, boxArea);
              set_values(featureID + 1 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x + box.width/3, box.y, box.width/3, box.height), -2.0f, type, boxArea);
              set_values(featureID + 2 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x + 2*box.width/3, box.y, box.width/3, box.height), 1.0f, type, boxArea);
              break;
            case 3:
              set_values(featureID + 0 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y, box.width, box.height/3), 1.0f, type, boxArea);
              set_values(featureID + 1 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y + box.height/3, box.width, box.height/3), -2.0f, type, boxArea);
              set_values(featureID + 2 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y + 2*box.height/3, box.width, box.height/3), 1.0f, type, boxArea);
              break;
            case 4:
              set_values(featureID + 0 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y, box.width/2, box.height/2), 1.0f, type, boxArea);
              set_values(featureID + 1 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x + box.width/2, box.y, box.width/2, box.height/2), -1.0f, type, boxArea);
              set_values(featureID + 2 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x, box.y + box.height/2, box.width/2, box.height/2), -1.0f, type, boxArea);
              set_values(featureID + 3 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x + box.width/2, box.y + box.height/2, box.width/2, box.height/2), 1.0f, type, boxArea);
              break;
            case 5:
              set_values(featureID + 0 * HAAR_FEATURE_COUNT, box, 1.0f, type, boxArea);
              set_values(featureID + 1 * HAAR_FEATURE_COUNT, cv::Rect_<float>(box.x + box.width/4, box.y + box.height/4, box.width/2, box.height/2), -4.0f, type, boxArea);
              break;
          }

          ++featureID;
        }
      }
    }
  }

  // Transfer the coordinates and weights across to the GPU.
  m_bottomsGPU = GPUVector<float>(m_bottomsCPU);
  m_leftsGPU = GPUVector<float>(m_leftsCPU);
  m_rightsGPU = GPUVector<float>(m_rightsCPU);
  m_topsGPU = GPUVector<float>(m_topsCPU);
  m_weightsGPU = GPUVector<float>(m_weightsCPU);
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

void HaarFeatureCalculator::calculate_features(cudaTextureObject_t frame, float *sampleData, int *keepSamples, size_t sampleCount,
                                               const cv::Rect_<float>& boundingBox, double *features, size_t offset, size_t stride,
                                               cudaTextureObject_t integralFrame, int threadsPerBlock) const
{
  calculate_haar_features(
    integralFrame,
    sampleData, keepSamples, sampleCount,
    boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height,
    m_bottomsGPU.get(), m_leftsGPU.get(), m_rightsGPU.get(), m_topsGPU.get(), m_weightsGPU.get(),
    features, offset, stride
  );
}

size_t HaarFeatureCalculator::feature_count() const
{
  return HAAR_FEATURE_COUNT;
}

bool HaarFeatureCalculator::needs_integral_frame() const
{
  return true;
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

void HaarFeatureCalculator::set_values(int index, const cv::Rect_<float>& minibox, float weight, int type, float boxArea)
{
  static float factors[] = { 255 * 1.0f / 2, 255 * 1.0f / 2, 255 * 2.0f / 3, 255 * 2.0f / 3, 255 * 1.0f / 2, 255 * 3.0f / 4 };
  m_bottomsCPU[index] = minibox.y + minibox.height;
  m_leftsCPU[index] = minibox.x;
  m_rightsCPU[index] = minibox.x + minibox.width;
  m_topsCPU[index] = minibox.y;
  m_weightsCPU[index] = weight / (factors[type] * boxArea);
}

}
