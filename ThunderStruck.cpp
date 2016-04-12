#include "stdafx.h"
#include "ThunderStruck.h"

ThunderStruck::ThunderStruck(cv::Mat& initframe, cv::Rect_<float>& initbox, CompositeFeatureCalculator_Ptr featureCalculator)
{
	m_tracker = new Tracker(initbox, featureCalculator);
	m_bIsRealTimeTracking = true;
	m_bIsFirstFrame = true;
	SetCudaDevice(TESLA_GPU);

	cv::Mat frame = initframe;
    m_tracker->update(initframe, m_bIsFirstFrame);

    // Provide the current frame and bounding box to the rendering thread.
    boost::lock_guard<boost::mutex> lock(g_mutex);
    initbox = m_tracker->get_current_bounding_box();

	m_bIsFirstFrame = false;
}

ThunderStruck::ThunderStruck(std::string sequencePath)
{
	m_bIsRealTimeTracking = false;
	m_sequencePath = sequencePath;

	// Read in the tracking sequence.
	g_sequence.reset(new Sequence(m_sequencePath));//带参数的reset()则类似相同形式的构造函数，原指针引用计数减1的同时改为管理另一个指针。

	SetCudaDevice(TESLA_GPU);
	m_bIsFirstFrame = true;
}

ThunderStruck::ThunderStruck()
{

}
ThunderStruck::~ThunderStruck(void)
{
	free(m_tracker);
}



void ThunderStruck::tracking()
{
	if (!m_bIsRealTimeTracking)
	{
		if (true == m_bIsFirstFrame)
		{
		   // Set the initial bounding box and frame.
			g_boundingBox = g_sequence->bounding_box(0);
			g_frame = g_sequence->frame(0);
			m_bIsFirstFrame = false;
		}
		
		run_tracking_offline();
	}

}

/**
 * \brief Runs the tracker (on a separate thread).
 */
void ThunderStruck::run_tracking_offline()
{
	timedelay.start();
	CompositeFeatureCalculator_Ptr featureCalculator(new CompositeFeatureCalculator);
 #if 0
	  featureCalculator->add_calculator(FeatureCalculator_CPtr(new RawFeatureCalculator));
#else
	   featureCalculator->add_calculator(FeatureCalculator_CPtr(new HaarFeatureCalculator));
	 #endif
#if 0
	featureCalculator->add_calculator(FeatureCalculator_CPtr(new RawFeatureCalculator));
#else
	featureCalculator->add_calculator(FeatureCalculator_CPtr(new HaarFeatureCalculator));
#endif

  Tracker tracker(g_boundingBox, featureCalculator);

  // Track the object indicated by the initial bounding box through the sequence.
//   double qualitySum = 0.0;
//   size_t qualityCount = 0;
  for(size_t i = 0, frameCount = g_sequence->frame_count(); i < frameCount; ++i)
  {
    // Update the tracker with the current frame from the sequence.
    cv::Mat frame = g_sequence->frame(i);
    tracker.update(g_sequence->frame(i), i == 0);

    // Provide the current frame and bounding box to the rendering thread.
    boost::lock_guard<boost::mutex> lock(g_mutex);
    g_boundingBox = tracker.get_current_bounding_box();
   /* g_frame = frame;
    g_frameIndex = i;*/

	// Make a colour version of the current frame for output purposes.
	cv::Mat3b output;
	if(frame.channels() == 1)
	{
		cvtColor(frame, frame, CV_GRAY2BGR);
	}
	else
	{
		output = frame.clone();
	}

	// Draw the current bounding box onto the output image.
	cv::rectangle(output, g_boundingBox.tl(), g_boundingBox.br(), CV_RGB(255,0,0));//tl()返回左上角点坐标，br()返回右下角点坐标。
	cv::imshow("Result",output);
	cv::waitKey(1);
  }

  float t = timedelay.end() / g_sequence->getFrameNum();
  std::cout<<"fps :"<<1000/t<<std::endl;
}


bool ThunderStruck::tracking(cv::Mat& frame, cv::Rect_<float>& box)
{
	m_tracker->update(frame, m_bIsFirstFrame);

	// Provide the current frame and bounding box to the rendering thread.
	boost::lock_guard<boost::mutex> lock(g_mutex);
	box = m_tracker->get_current_bounding_box();
	return true;
}