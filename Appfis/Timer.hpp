/** @file Timer.hpp
*  @brief Header containing the Timer class
*
*  This contains the subroutines and eventually any
*  macros, constants, etc. needed for Timer class
*
*  @author Md Bulbul Sharif
*/


#ifndef TIMER_H
#define TIMER_H


#include <cmath>
#include <vector>
#include <map>


#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)

#include <sys/timeb.h>
#include <sys/types.h>
#include <winsock2.h>
#define __need_clock_t

#include <time.h>

struct tms
{
	clock_t tms_utime;
	clock_t tms_stime;

	clock_t tms_cutime;
	clock_t tms_cstime;
};

clock_t times(struct tms* __buffer);
typedef long long suseconds_t;

int gettimeofday(struct timeval* t, void* timezone)
{
	struct _timeb timebuffer;
	_ftime(&timebuffer);
	t->tv_sec = static_cast<long>(timebuffer.time);
	t->tv_usec = 1000 * timebuffer.millitm;
	return 0;
}

clock_t times(struct tms* __buffer) {

	__buffer->tms_utime = clock();
	__buffer->tms_stime = 0;
	__buffer->tms_cstime = 0;
	__buffer->tms_cutime = 0;
	return __buffer->tms_utime;
}

#else
#include <sys/time.h>
#include <unistd.h>
#endif


namespace APPFIS
{

	typedef unsigned long long ull;    /**< Custom data type to hold large number. */
	const int TIMER_NSECS = 0;    /**< To use nano second in Timer. */
	const int TIMER_SECS = 1;    /**< To use second in Timer. */


	struct StringCompare	/**< Structure to compare to string. */
	{

		struct CharCompare	/**< Structure to compare to char. */
		{

			/** @brief It compare to char. If first char is less than second char, it returns true.
			*
			*  @param c1 First char
			*  @param c2 Second char
			*  @return True or False
			*/
			bool operator() (const unsigned char& c1, const unsigned char& c2) const
			{
				return tolower(c1) < tolower(c2);
			}

		};


		/** @brief It compare to string. If first string is less than second string, it returns true. Compare happens one by one char.
		*
		*  @param s1 First string
		*  @param s2 Second string
		*  @return True or False
		*/
		bool operator() (const std::string& s1, const std::string& s2) const
		{
			return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(), CharCompare());
		}

	};


	class Timer	/**< Custom timer class to compute time for different operation. */
	{

	public:

		/** @brief Constructor
		*
		*/
		Timer();


		/** @brief It starts a custom timer.
		*
		*  @param category Timer name
		*/
		void Start(std::string category);


		/** @brief It stops a custom timer.
		*
		*  @param category Timer name
		*/
		void Stop(std::string category);


		/** @brief It resets all timer.
		*
		*/
		void Reset();


		/** @brief It calculates total time of every timer.
		*
		*  @return Time value
		*/
		double GetTotalTime();


		/** @brief It calculates time value of a specific timer.
		*
		*  @param category Timer name
		*  @return Time value
		*/
		double GetCustomTime(std::string category);


		/** @brief It calculates current date.
		*
		*  @return Current date
		*/
		std::string GetCurrentDate();


		/** @brief It calculates host name.
		*
		*  @return Host name
		*/
		std::string GetHostname();


		/** @brief It helps to add a new timer.
		*
		*  @param category Timer name
		*  @return Timer id
		*/
		int AddNewTimer(std::string category);


		/** @brief It calculates billion lattice updates per second.
		*
		*  @param category Timer name
		*  @param size Total size of the grid
		*  @param iter Number of iteration
		*  @return BLUPS
		*/
		double CalculateBLUPS(std::string category, int size, int iter);


	private:

		int _units;	/**< Time units of a timer */
		double _unitFactor;	/**< Time unit factor of a timer */
		std::vector<timeval> _allTimers;	/**< Vector that contains all timer information */
		std::map<std::string, int, StringCompare> _timeCats;	/**< Timer name, size mapping */
		std::vector<ull> _times;	/**< Vector to contains timers time value */


		/** @brief It sets time of a timer.
		*
		*  @param category Timer name
		*  @param time Timer time value
		*/
		void UpdateTime(std::string category, ull time);


		/** @brief It calculates index id of a timer.
		*
		*  @param category Timer name
		*  @return Timer id
		*/
		int GetCatIndex(std::string category);


		/** @brief It initializes timer object.
		*
		*/
		void Init();


		/** @brief It updates time of a timer.
		*
		*  @param timer Timer id
		*  @param time Timer time value
		*/
		void UpdateTime(int timer, ull time);


		/** @brief It converts time value to floating point number.
		*
		*  @param time Timer time value
		*  @return Time value
		*/
		double Convert(ull time);

	};


	Timer::Timer() : _units(TIMER_SECS)
	{
		this->Init();
	}


	void Timer::Init()
	{
		this->_unitFactor = 1;
		if (this->_units == TIMER_SECS)
		{
			this->_unitFactor = 0.000001;
		}
		else if (this->_units == TIMER_NSECS)
		{
			this->_unitFactor = 1000;
		}
	}


	int Timer::AddNewTimer(std::string category)
	{
		int ncats = static_cast<int>(this->_timeCats.size());
		std::map<std::string, int, StringCompare>::iterator it = this->_timeCats.find(category);
		if (it == this->_timeCats.end())
		{
			timeval ntval;
			ull newcat_time = 0;

			this->_timeCats.insert(std::pair<std::string, int>(category, ncats));
			this->_times.push_back(newcat_time);
			this->_allTimers.push_back(ntval);
		}

		return static_cast<int>(this->_timeCats.size() - 1);
	}


	int Timer::GetCatIndex(std::string category)
	{
		int catidx;
		std::map<std::string, int, StringCompare>::iterator it = this->_timeCats.find(category);

		if (it != this->_timeCats.end())
		{
			catidx = it->second;
		}
		else
		{
			this->AddNewTimer(category);
			catidx = this->AddNewTimer(category);
		}

		return catidx;
	}


	void Timer::UpdateTime(int catid, ull time)
	{
		if (catid < (int)this->_times.size())
		{
			this->_times[catid] += time;
		}
	}


	void Timer::UpdateTime(std::string category, ull time)
	{
		this->_times[this->GetCatIndex(category)] += time;
	}


	double Timer::Convert(ull time)
	{
		return (double)time * this->_unitFactor;
	}


	void Timer::Reset()
	{
		this->_allTimers.clear();
		this->_times.clear();

		this->Init();
	}


	void Timer::Start(std::string category)
	{
		timeval* t = &this->_allTimers[this->GetCatIndex(category)];

		gettimeofday(t, NULL);
	}


	void Timer::Stop(std::string category)
	{
		timeval t1, t2;
		int catidx = this->GetCatIndex(category);
		gettimeofday(&t2, NULL);
		t1 = this->_allTimers[catidx];

		ull time_usecs = ((((ull)t2.tv_sec * 1000000) + t2.tv_usec) - (((ull)t1.tv_sec * 1000000) + t1.tv_usec));

		this->UpdateTime(category, time_usecs);
	}


	double Timer::GetCustomTime(std::string category)
	{
		return this->Convert(this->_times[this->GetCatIndex(category)]);
	}


	double Timer::GetTotalTime()
	{
		double total = 0.0;
		std::map<std::string, int>::iterator it = this->_timeCats.begin();

		for (it = this->_timeCats.begin(); it != this->_timeCats.end(); it++)
		{
			total += this->GetCustomTime(it->first);
		}

		return total;
	}


	std::string Timer::GetCurrentDate()
	{
		time_t rawtime;
		struct tm* timeinfo;
		char buffer[24];

		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer, 24, "%D", timeinfo);

		return std::string(buffer);
	}


	std::string Timer::GetHostname()
	{
		char hostname[1024];
		hostname[1023] = '\0';

#ifdef _WIN32
		/*if (gethostname(hostname, 1023) != 0)
		{
			std::cerr << "There is an error" << std::endl;
		}*/

		std::cerr << "This is inside Timer.hpp. To use this option you need to add Ws2_32.lib as dependency and uncomment previous lines." << std::endl;

#else
		gethostname(hostname, 1023);
#endif

		return std::string(hostname);
	}


	double Timer::CalculateBLUPS(std::string category, int size, int iter)
	{
		double t = this->GetCustomTime(category);
		return ((double)size * iter) / (t * std::pow(10, 9));
	}

}


#endif