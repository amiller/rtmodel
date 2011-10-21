/*
 *  Copyright 2008-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#ifndef __TIMER_H__
#define __TIMER_H__

#include <cuda.h>

#ifdef _WIN32
#include <windows.h>


struct cpu_timer
{
	LARGE_INTEGER	m_startTime, m_endTime;

    cpu_timer() { }
    ~cpu_timer() { }

    void start() { QueryPerformanceCounter(&m_startTime); }
    void stop()  { cudaThreadSynchronize(); QueryPerformanceCounter(&m_endTime);}

    float elapsed_sec() { 
        LARGE_INTEGER diff;
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        diff.QuadPart = m_endTime.QuadPart - m_startTime.QuadPart;
        return (float)diff.QuadPart / (float)freq.QuadPart;
    }

    float elapsed_ms()
    {
        return elapsed_sec() * 1000.0f;
    }

};

#else

#include <sys/time.h>


struct cpu_timer 
{
  double _start_time, _end_time;

  cpu_timer() { }
  ~cpu_timer() { }

  void start() { 
    struct timeval t;    
    gettimeofday(&t, 0); 
    _start_time = t.tv_sec + (1e-6 * t.tv_usec);
  }
  void stop()  { 
    struct timeval t;    
    gettimeofday(&t, 0); 
    _end_time = t.tv_sec + (1e-6 * t.tv_usec);
  }

  float elapsed_sec() { 
    return (float) (_end_time - _start_time);
  }

  float elapsed_ms()
  {
    return (float) 1000 * (_end_time - _start_time);
  }
};



#endif

struct gpu_timer
{
    cudaEvent_t e_start, e_stop;

    gpu_timer() { cudaEventCreate(&e_start);  cudaEventCreate(&e_stop); }
    ~gpu_timer() { cudaEventDestroy(e_start); cudaEventDestroy(e_stop); }

    void start() { cudaEventRecord(e_start, 0); }
    void stop()  { cudaEventRecord(e_stop, 0); }

    float elapsed_ms()
    {
        cudaEventSynchronize(e_stop);
        float ms;
        cudaEventElapsedTime(&ms, e_start, e_stop);
        return ms;
    }

    float elapsed_sec() { return elapsed_ms() / 1000.0f; }
};


#endif
