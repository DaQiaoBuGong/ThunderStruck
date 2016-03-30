#pragma once
#include <windows.h>
class CTimeDelay
{
public:
	CTimeDelay();
	~CTimeDelay(void);
protected:
	//¼ÆÊ±Æ÷
	LARGE_INTEGER litmp; 
	LONGLONG QPart1,QPart2; 
	double dfMinus, dfFreq, dfTim; 
public:
	void timeDelay(float ms);
	void start();
	float end();
};

