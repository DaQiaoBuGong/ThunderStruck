#pragma once

#include "TimeDelay.h"
#include <iostream>
#include <string.h>
#include <boost/thread.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <opencv2/opencv.hpp>
#include <process.h>


#include "Sequence.h"   //参数文件
#include "Tracker.h"
#include "CompositeFeatureCalculator.h"
#include "HaarFeatureCalculator.h"
//#include "RawFeatureCalculator.h"
#include "GeomUtil.h"

const int TESLA_GPU = 0;

using namespace thunderstruck;

class ThunderStruck
{
public:
	ThunderStruck(void);
	ThunderStruck(std::string sequencePath);
	ThunderStruck(cv::Mat& initframe, cv::Rect_<float>& initbox, CompositeFeatureCalculator_Ptr featureCalculator);

	~ThunderStruck(void);

private:

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

	//计时
	CTimeDelay timedelay;

	//跟踪标志位
	bool m_bIsRealTimeTracking;
	bool m_bIsFirstFrame;


	//Trackingkernel
	void run_tracking_online();
	void run_tracking_offline();
	std::string m_sequencePath;

	Tracker* m_tracker;

public:
	void tracking();

	//实时跟踪
	void tracking(cv::Mat& frame);


	void SetFirstFrame(bool firstFrame){m_bIsFirstFrame = firstFrame;};
	bool GetFirstFrame(){return m_bIsFirstFrame;};

	void SetRealTimeTracking(bool RealTime){m_bIsRealTimeTracking = RealTime;};
	bool GetRealTimeTracking(){return m_bIsRealTimeTracking;};


	bool tracking(cv::Mat& frame, cv::Rect_<float>& box);

};

