// test_gpu_opencv2.4.12.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "TimeDelay.h"
#include <iostream>
#include <string.h>
#include <boost/thread.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <opencv2/opencv.hpp>

#include "Sequence.h"   //参数文件
#include "Tracker.h"
#include "CompositeFeatureCalculator.h"
#include "HaarFeatureCalculator.h"
#include "RawFeatureCalculator.h"
#include "GeomUtil.h"

using namespace thunderstruck;

CTimeDelay timedelay;
//#################### GLOBAL VARIABLES ####################

/** The bounding box for the current tracking frame (shared between the two threads). */
cv::Rect_<float> g_boundingBox;

/** The current tracking frame (shared between the two threads). */
boost::optional<cv::Mat> g_frame;

/** The index of the current tracking frame (shared between the two threads). */
size_t g_frameIndex;

/** The mutex used to synchronise the two threads. */
boost::mutex g_mutex;

/** The tracking sequence (shared between the two threads, but immutable once constructed). */
boost::shared_ptr<Sequence> g_sequence;


//#################### GLOBAL FUNCTIONS ####################

/**
 * \brief Renders the tracker's current frame and bounding box.
 */
bool render_frame()
{
  // Get the current bounding box and frame.
  cv::Rect boundingBox;
  boost::optional<cv::Mat> frame;
  size_t frameIndex;

  {
    boost::lock_guard<boost::mutex> lock(g_mutex);
    boundingBox = g_boundingBox; // note: we truncate the coordinates of the bounding box to integers here
    frame = g_frame;
    frameIndex = g_frameIndex;

#if 0
    // To step the tracker one frame at a time, we wait for a key here whilst holding the mutex (to stop the tracking).
    cv::waitKey(0);
#endif
  }

  // If we've reached the end of the tracking sequence, exit.
  if(!frame) return false;

  // Make a colour version of the current frame for output purposes.
  cv::Mat3b output;
  if(frame->channels() == 1)
  {
    cv::Mat1b channels[] = { *frame, *frame, *frame };
    cv::merge(channels, 3, output);
  }
  else
  {
    output = frame->clone();
  }

  // Draw the current bounding box onto the output image.
  cv::rectangle(output, boundingBox.tl(), boundingBox.br(), CV_RGB(255,0,0));

#if 0
  // Draw the most recent ground truth bounding box onto the output image.
  cv::Rect gtBoundingBox;
  for(int i = frameIndex; i >= 0; --i)
  {
    gtBoundingBox = g_sequence->bounding_box(i); // note: we truncate the coordinates of the bounding box to integers here
    if(gtBoundingBox.width > 0) break;
  }
  cv::rectangle(output, gtBoundingBox.tl(), gtBoundingBox.br(), CV_RGB(0,255,0));
#endif

  // Show the output image.
  cv::imshow("stricken", output);
  //cv::imwrite("test.bmp", output);
  cv::waitKey(1);

  return true;
}

/**
 * \brief Runs the tracker (on a separate thread).
 */
void run_tracking()
{
	timedelay.start();
  // Set up the tracker.
  CompositeFeatureCalculator_Ptr featureCalculator(new CompositeFeatureCalculator);
#if 0
  featureCalculator->add_calculator(FeatureCalculator_CPtr(new RawFeatureCalculator));
#else
  featureCalculator->add_calculator(FeatureCalculator_CPtr(new HaarFeatureCalculator));
#endif
  Tracker tracker(g_boundingBox, featureCalculator);

  // Track the object indicated by the initial bounding box through the sequence.
  double qualitySum = 0.0;
  size_t qualityCount = 0;
  for(size_t i = 0, frameCount = g_sequence->frame_count(); i < frameCount; ++i)
  {
    // Update the tracker with the current frame from the sequence.
    cv::Mat frame = g_sequence->frame(i);
    tracker.update(g_sequence->frame(i), i == 0);

    // Update the record of tracking quality.
    const cv::Rect_<float>& groundTruthBoundingBox = g_sequence->bounding_box(i);
    if(groundTruthBoundingBox.width != 0)
    {
      qualitySum += GeomUtil::compute_overlap(tracker.get_current_bounding_box(), groundTruthBoundingBox);
      ++qualityCount;
    }

    if(qualityCount != 0)
    {
      //std::cout << "RUNNING TRACKING QUALITY: " << qualitySum / qualityCount << '\n';
    }

    // Provide the current frame and bounding box to the rendering thread.
    boost::lock_guard<boost::mutex> lock(g_mutex);
    g_boundingBox = tracker.get_current_bounding_box();
    g_frame = frame;
    g_frameIndex = i;
  }

  //std::cout << "OVERALL TRACKING QUALITY: " << qualitySum / qualityCount << '\n';

  {
    // Clear the current frame when we get to the end of the tracking sequence.
    boost::lock_guard<boost::mutex> lock(g_mutex);
    g_frame = boost::none;
  }
  float t = timedelay.end() / g_sequence->getFrameNum();
  std::cout<<"fps :"<<1000/t<<std::endl;
  
}

int _tmain(int argc,char **argv)
{
	
	// Determine the sequence path.
	std::string sequencePath;
// 	switch(argc)
// 	{
// 	case 1:
// 		// Use the default sequence path. Note that we're in thunderstruck/build/bin/stricken,
// 		// so this points to thunderstruck/sequences/girl.
// 		sequencePath = "Girl/";
// 		break;
// 	case 2:
// 		// Construct a sequence path from the name specified on the command line.
// 		sequencePath = "../../../sequences/" + std::string(argv[1]) + "/";
// 		break;
// 	default:
// 		std::cout << "Usage: stricken [sequence name]" << std::endl;
// 		return 0;
// 	}

	sequencePath = "data/Car4/img/";

	// Read in the tracking sequence.
	g_sequence.reset(new Sequence(sequencePath));//带参数的reset()则类似相同形式的构造函数，原指针引用计数减1的同时改为管理另一个指针。

	// Choose the best CUDA device.
	findCudaDevice(argc, const_cast<const char**>(argv));

	

	// Set the initial bounding box and frame.
	g_boundingBox = g_sequence->bounding_box(0);
	g_frame = g_sequence->frame(0);

	// Set up the rendering window and render the first frame.
	cv::namedWindow("stricken", CV_WINDOW_AUTOSIZE);
	render_frame();

	// Start the tracker and enter the rendering loop.
	boost::thread trackingThread(&run_tracking);
	trackingThread.detach();
	while(render_frame());

	cv::waitKey(0);
	 
	return 0;
}

