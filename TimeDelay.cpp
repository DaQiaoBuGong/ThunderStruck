#include "StdAfx.h"
#include "TimeDelay.h"


CTimeDelay::CTimeDelay()
{
	dfMinus = 0;
	dfTim = 0;
	QueryPerformanceFrequency(&litmp); 
	dfFreq = litmp.QuadPart;
	QPart1 = 0;
	QPart2 = 0;
}


CTimeDelay::~CTimeDelay(void)
{
}

void CTimeDelay::timeDelay(float ms)
{
	float delayms = ms / 1000;
	QueryPerformanceCounter(&litmp);
	QPart1 = litmp.QuadPart;

	//延时
	do 
	{ 
		QueryPerformanceCounter(&litmp); 
		QPart2 = litmp.QuadPart;//获得中止值 
		dfMinus = (double)(QPart2-QPart1); 
		dfTim = dfMinus / dfFreq;// 获得对应的时间值，单位为秒
	}while(dfTim < delayms);
}

void CTimeDelay::start()
{
	QueryPerformanceCounter(&litmp);
	QPart1 = litmp.QuadPart;
}

float CTimeDelay::end()
{
	QueryPerformanceCounter(&litmp); 
	QPart2 = litmp.QuadPart;//获得中止值 
	dfMinus = (double)(QPart2-QPart1); 
	dfTim = dfMinus / dfFreq;// 获得对应的时间值，单位为秒
	dfTim = dfTim * 1000;
	
	return dfTim;
}
