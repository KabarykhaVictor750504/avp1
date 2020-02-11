#pragma once
#include <chrono>

class Timer
{
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> startTimepoint;

public:
	Timer()
	{
		startTimepoint = std::chrono::high_resolution_clock::now();
	}

	//microseconds
	long long TimeFromBegin()
	{
		auto endTimepoint = std::chrono::high_resolution_clock::now();

		auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimepoint).time_since_epoch().count();
		auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

		return end - start;
	}
};