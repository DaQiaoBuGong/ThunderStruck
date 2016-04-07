#pragma once

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
//#include "RawFeatureCalculator.h"
#include "GeomUtil.h"

using namespace thunderstruck;

class ThunderStruck
{
public:
	ThunderStruck(void);
	~ThunderStruck(void);
	std::string sequencePath;

private:
	bool render_frame();
	void run_tracking();

public:
	void tracking();
};

