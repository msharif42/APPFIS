/**
 *
 * Serial version of FDTD3d
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef FDTD3D_SERIAL_HPP
#define FDTD3D_SERIAL_HPP


#include <iostream>
#include <limits>
#include "../../Appfis/Timer.hpp"
#include "../../Appfis/Utils.hpp"
using namespace std;


void FDTD3d(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int dimZ, int radius, float* data);
void Simulate(int dimX, int dimY, int dimZ, float coeff, int radius, float* cur, float* next);
int Index1D(int i, int j, int k);
int strideX;
int strideY;


void FDTD3d(int argc, char* argv[])
{
	APPFIS::Timer timer = APPFIS::Timer();
	timer.Start("TOTAL");

	int dimX = 64;
	int dimY = 64;
	int dimZ = 64;
	int iteration = 10;
	int radius = 4;
	float coeff = 0.1f;

	if (APPFIS::CheckArgument(argc, argv, "dimx")) {
		dimX = APPFIS::GetArgument(argc, argv, "dimx");
	}
	if (APPFIS::CheckArgument(argc, argv, "dimy")) {
		dimY = APPFIS::GetArgument(argc, argv, "dimy");
	}
	if (APPFIS::CheckArgument(argc, argv, "dimz")) {
		dimZ = APPFIS::GetArgument(argc, argv, "dimz");
	}
	if (APPFIS::CheckArgument(argc, argv, "iteration")) {
		iteration = APPFIS::GetArgument(argc, argv, "iteration");
	}

	strideX = (dimX + 2 * radius);
	strideY = (dimY + 2 * radius);
	float* grid = new float[(dimX + 2 * radius) * (dimY + 2 * radius) * (dimZ + 2 * radius)]();
	float* next = new float[(dimX + 2 * radius) * (dimY + 2 * radius) * (dimZ + 2 * radius)]();

	InitialData(dimX, dimY, dimZ, radius, grid);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		Simulate(dimX, dimY, dimZ, coeff, radius, grid, next);

		float* tmp = grid;
		grid = next;
		next = tmp;
	}
	timer.Stop("COMPUTE");

	delete[] grid;
	delete[] next;

	timer.Stop("TOTAL");

	double totalTime = timer.GetCustomTime("TOTAL");
	double computeTime = timer.GetCustomTime("COMPUTE");
	std::cout << "Total: " << totalTime << std::endl;
	std::cout << "Compute: " << computeTime << std::endl;
	std::cout << "GFLOPS: " << 25 * timer.CalculateBLUPS("COMPUTE", dimX * dimY * dimZ, iteration) << std::endl;
}


void InitialData(int dimX, int dimY, int dimZ, int radius, float* data)
{
	srand(0);
	const float lowerBound = 0.0f;
	const float upperBound = 1.0f;

	for (int z = 0; z < dimZ; z++)
	{
		for (int y = 0; y < dimY; y++)
		{
			for (int x = 0; x < dimX; x++)
			{
				float val = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
				data[(x + radius) + (dimX + 2 * radius) * (y + radius) + (dimX + 2 * radius) * (dimY + 2 * radius) * (z + radius)] = val;
			}
		}
	}
}


void Simulate(int dimX, int dimY, int dimZ, float coeff, int radius, float* cur, float* next)
{
	for (int k = radius; k < dimZ + radius; k++)
	{
		for (int j = radius; j < dimY + radius; j++)
		{
			for (int i = radius; i < dimX + radius; i++)
			{
				float val = cur[Index1D(i, j, k)];

				for (int a = 1; a <= radius; a++)
				{
					val += (cur[Index1D(i - a, j, k)] + cur[Index1D(i + a, j, k)])
						+ (cur[Index1D(i, j - a, k)] + cur[Index1D(i, j + a, k)])
						+ (cur[Index1D(i, j, k - a)] + cur[Index1D(i, j, k + a)]);
				}

				val *= coeff;

				next[Index1D(i, j, k)] = val;
			}
		}
	}
}


int Index1D(int i, int j, int k)
{
	return i + j * strideX + k * strideX * strideY;
}


#endif