/**
 * thunderstruck/tracker: Tracker.cpp
 */
#include "stdafx.h"
#include "Tracker.h"

#include <stdexcept>

#include "SampleFilterer.h"
#include "Sampler.h"
#include "FeatureCalculator.h"
#include "Timing.h"

namespace thunderstruck {

//#################### CONSTRUCTORS ####################

Tracker::Tracker(const cv::Rect_<float>& initialBoundingBox, const FeatureCalculator_CPtr& featureCalculator)
: m_curBoundingBox(initialBoundingBox), m_featureCalculator(featureCalculator), m_svm(100.0, initialBoundingBox, featureCalculator)
{
  // Determine the sample points for tracking. Note that the
  // points are specified relative to the origin, which allows
  // us to reuse the same points from one frame to the next.
  const int trackingRadius = 30;
  std::vector<cv::Vec2f> trackingSamples = Sampler::make_pixel_samples(trackingRadius);
  m_trackingSampleCount = trackingSamples.size();

  // Transfer the sample points across to the GPU.
  m_trackingSampleData = Sampler::cpu_to_gpu(trackingSamples);
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

const cv::Rect_<float>& Tracker::get_current_bounding_box() const
{
  return m_curBoundingBox;
}

void Tracker::update(const cv::Mat& frame, bool initialFrame)
{
#if 1
  // Output timing information for the frame.
  static boost::chrono::high_resolution_clock::time_point t0 = boost::chrono::high_resolution_clock::now();
  boost::chrono::high_resolution_clock::time_point t1 = boost::chrono::high_resolution_clock::now();
  boost::chrono::milliseconds dur = boost::chrono::duration_cast<boost::chrono::milliseconds>(t1 - t0);
  std::cout << "\nTime since last frame: " << dur << '\n';
  if(dur.count() > 0)
  {
    float fps = 1000.0f / dur.count();
    std::cout << "Frames per second: " << fps << '\n';
  }
  t0 = t1;

  static boost::chrono::milliseconds totalDur(0);
  static int frameCount = 0;
  totalDur += dur;
  ++frameCount;
  boost::chrono::milliseconds avgDur = totalDur / frameCount;
  if(avgDur.count() > 0)
  {
    float fps = 1000.0f / avgDur.count();
    std::cout << "Average frames per second: " << fps << '\n';
  }
#endif

  // Convert the current frame to greyscale as necessary.
  cv::Mat1b greyscaleFrame;
  if(frame.channels() == 1)
  {
    greyscaleFrame = frame;
  }
  else if(frame.channels() == 3)
  {
    cv::cvtColor(frame, greyscaleFrame, CV_BGR2GRAY);
  }
  else
  {
    throw std::runtime_error("The current frame has an unsupported number of channels");
  }

  // Calculate any supporting images (e.g. the integral image) that are needed for feature computation on the GPU.
  cv::Mat1i integralFrame = cv::Mat1i::zeros(greyscaleFrame.rows + 1, greyscaleFrame.cols + 1);
  cv::integral(greyscaleFrame, integralFrame);

  // Transfer the relevant images across to the GPU.
  GPUTexture1b gpuFrame(greyscaleFrame);
  boost::optional<GPUTexture1i> gpuIntegralFrame;
  if(m_featureCalculator->needs_integral_frame())
  {
    gpuIntegralFrame.reset(GPUTexture1i(integralFrame));
  }

#if 0
  // If this is the initial frame, simply update the SVM using the frame data.
  // If not, attempt to predict a new bounding box and then update the SVM.
  // (Note that we avoid updating the SVM if prediction fails.)
  if(initialFrame || predict_bounding_box(gpuFrame, gpuIntegralFrame))
  {
    m_svm.update(gpuFrame, m_curBoundingBox, gpuIntegralFrame);
  }
#else
  // Note: This is how Struck currently works - we can use this version for comparison purposes when debugging.

  // If this is the initial frame, do an initial update of the SVM using the frame data.
  if(initialFrame)
  {
    m_svm.update(gpuFrame, m_curBoundingBox, gpuIntegralFrame);
  }

  // Attempt to predict a new bounding box and then update the SVM.
  // (Note that we avoid updating the SVM if prediction fails.)
  if(predict_bounding_box(gpuFrame, gpuIntegralFrame))
  {
    m_svm.update(gpuFrame, m_curBoundingBox, gpuIntegralFrame);
  }
#endif
}

//#################### PRIVATE MEMBER FUNCTIONS ####################

bool Tracker::predict_bounding_box(const GPUTexture1b& frame, const boost::optional<GPUTexture1i>& integralFrame)
{
  // Determine which tracking sample bounding boxes fall entirely within the frame.
  GPUVector<int> keepSamples(m_trackingSampleCount);

  {
    CUDA_TIME(filter_samples(
      m_trackingSampleData.get(), m_trackingSampleCount, keepSamples.get(),
      m_curBoundingBox.x, m_curBoundingBox.y, m_curBoundingBox.width, m_curBoundingBox.height,
      frame.width(), frame.height()
    ), microseconds);
    std::cout << "Time to filter samples for prediction: " << dur << '\n';
  }

  // Calculate feature vectors for those bounding boxes.
  GPUVector<double> sampleFeatures(m_trackingSampleCount * m_featureCalculator->feature_count());

  {
    CUDA_TIME(m_featureCalculator->calculate_features(
      frame.get(), m_trackingSampleData.get(), keepSamples.get(), m_trackingSampleCount,
      m_curBoundingBox, sampleFeatures.get(), integralFrame ? integralFrame->get() : 0
    ), microseconds);
    std::cout << "Time to calculate features for prediction: " << dur << '\n';
  }

  // Evaluate the SVM on those bounding boxes and pick the best one (if any) as the predicted bounding box for this frame.
  {
    CUDA_TIME({
      GPUVector<double> sampleResultsGPU(m_trackingSampleCount);
      m_svm.evaluate(sampleFeatures, keepSamples, sampleResultsGPU);

      std::vector<int> keepSamplesCPU = keepSamples.to_cpu();
      std::vector<double> sampleResultsCPU = sampleResultsGPU.to_cpu();
      double bestSampleResult = INT_MIN;
      size_t bestSampleIndex = INT_MAX;
      for(size_t i = 0, size = sampleResultsCPU.size(); i < size; ++i)
      {
        if(keepSamplesCPU[i] != 0 && sampleResultsCPU[i] > bestSampleResult)
        {
          bestSampleResult = sampleResultsCPU[i];
          bestSampleIndex = i;
        }
      }

      if(bestSampleIndex != INT_MAX)
      {
        float xOffset = m_trackingSampleData.get(bestSampleIndex * 2);
        float yOffset = m_trackingSampleData.get(bestSampleIndex * 2 + 1);
        m_curBoundingBox.x += xOffset;
        m_curBoundingBox.y += yOffset;
      }
    }, microseconds);
    std::cout << "Time to pick the best bounding box: " << dur << '\n';
  }

  return true;
}

}
