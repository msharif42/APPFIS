/**
 *
 * Serial version of Game Of Life
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef GAME_SERIAL_HPP
#define GAME_SERIAL_HPP


#include <iostream>
#include "../../Appfis/Timer.hpp"
#include "../../Appfis/Utils.hpp"
using namespace std;


void GameOfLife(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int* data);
void Simulate(int dimX, int dimY, int* cur, int* next);
int Index1D(int i, int j);
int stride;


void GameOfLife(int argc, char* argv[])
{
	APPFIS::Timer timer = APPFIS::Timer();
	timer.Start("TOTAL");

	int dimX = 64;
	int dimY = 64;
	int iteration = 10;
	bool outputFlag = false;

	if (APPFIS::CheckArgument(argc, argv, "dimx")) {
		dimX = APPFIS::GetArgument(argc, argv, "dimx");
	}
	if (APPFIS::CheckArgument(argc, argv, "dimy")) {
		dimY = APPFIS::GetArgument(argc, argv, "dimy");
	}
	if (APPFIS::CheckArgument(argc, argv, "iteration")) {
		iteration = APPFIS::GetArgument(argc, argv, "iteration");
	}
	outputFlag = APPFIS::CheckArgument(argc, argv, "output");

	stride = (dimX + 2);
	int* grid = new int[(dimX + 2) * (dimY + 2)]();
	int* next = new int[(dimX + 2) * (dimY + 2)]();

	InitialData(dimX, dimY, grid);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		Simulate(dimX, dimY, grid, next);

		int* tmp = grid;
		grid = next;
		next = tmp;
	}
	timer.Stop("COMPUTE");

	if (outputFlag)
	{
		int liveCount = 0;
		for (int j = 1; j <= dimY; j++)
		{
			for (int i = 1; i <= dimX; i++)
			{
				liveCount += grid[i + j * (dimX + 2)];
			}
		}
		std::cout << "Live Count: " << liveCount << std::endl;
	}

	delete[] grid;
	delete[] next;

	timer.Stop("TOTAL");

	double totalTime = timer.GetCustomTime("TOTAL");
	double computeTime = timer.GetCustomTime("COMPUTE");
	std::cout << "Total: " << totalTime << std::endl;
	std::cout << "Compute: " << computeTime << std::endl;
	std::cout << "GFLOPS: " << 7 * timer.CalculateBLUPS("COMPUTE", dimX * dimY, iteration) << std::endl;
}


void InitialData(int dimX, int dimY, int* data)
{
	srand(0);
	for (int j = 1; j <= dimY; j++)
	{
		for (int i = 1; i <= dimX; i++)
		{
			data[i + j * (dimX + 2)] = rand() % 2;
		}
	}
}


void Simulate(int dimX, int dimY, int* cur, int* next)
{
	for (int j = 1; j <= dimY; j++)
	{
		cur[Index1D(0, j)] = cur[Index1D(dimX, j)];
		cur[Index1D(dimX + 1, j)] = cur[Index1D(1, j)];
	}

	for (int i = 0; i <= dimX + 1; i++)
	{
		cur[Index1D(i, 0)] = cur[Index1D(i, dimY)];
		cur[Index1D(i, dimY + 1)] = cur[Index1D(i, 1)];
	}

	for (int j = 1; j <= dimY; j++)
	{
		for (int i = 1; i <= dimX; i++)
		{
			int count = cur[Index1D(i - 1, j - 1)] + cur[Index1D(i, j - 1)] + cur[Index1D(i + 1, j - 1)]
				+ cur[Index1D(i - 1, j)] + cur[Index1D(i + 1, j)]
				+ cur[Index1D(i - 1, j + 1)] + cur[Index1D(i, j + 1)] + cur[Index1D(i + 1, j + 1)];

			if (cur[Index1D(i, j)] == 1)
			{
				if (count < 2 || count > 4) next[Index1D(i, j)] = 0;
			}
			else
			{
				if (count == 3) next[Index1D(i, j)] = 1;
			}
		}
	}
}


int Index1D(int i, int j)
{
	return i + j * stride;
}


#endif