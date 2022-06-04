/**
 *
 * Serial version of Heat Diffusion
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef HEAT_SERIAL_HPP
#define HEAT_SERIAL_HPP


#include <iostream>
#include <limits>
#include "../../Appfis/Timer.hpp"
#include "../../Appfis/Utils.hpp"
using namespace std;


void HeatDiffusion(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int dimZ, double* data);
void Simulate(int dimX, int dimY, int dimZ, double cx, double cy, double cz, double* cur, double* next);
int Index1D(int i, int j, int k);
int strideX;
int strideY;


void HeatDiffusion(int argc, char* argv[])
{
	APPFIS::Timer timer = APPFIS::Timer();
	timer.Start("TOTAL");

	int dimX = 64;
	int dimY = 64;
	int dimZ = 64;
	int iteration = 10;
	double dt = 0.1;
	double dc = 0.01;
	double dx, dy, dz;
	dx = dy = dz = 0.1;
	double cx = (dt * dc) / pow(dx, 2);
	double cy = (dt * dc) / pow(dy, 2);
	double cz = (dt * dc) / pow(dz, 2);

	bool outputFlag = false;

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
	outputFlag = APPFIS::CheckArgument(argc, argv, "output");

	strideX = (dimX + 2);
	strideY = (dimY + 2);
	double* grid = new double[(dimX + 2) * (dimY + 2) * (dimZ + 2)]();
	double* next = new double[(dimX + 2) * (dimY + 2) * (dimZ + 2)]();

	InitialData(dimX, dimY, dimZ, grid);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		Simulate(dimX, dimY, dimZ, cx, cy, cz, grid, next);

		double* tmp = grid;
		grid = next;
		next = tmp;
	}
	timer.Stop("COMPUTE");

	if (outputFlag)
	{
		double maxHeat = (std::numeric_limits<double>::min)();
		double minHeat = (std::numeric_limits<double>::max)();
		for (int k = 1; k <= dimZ; k++)
		{
			for (int j = 1; j <= dimY; j++)
			{
				for (int i = 1; i <= dimX; i++)
				{
					int index = i + (dimX + 2) * j + (dimX + 2) * (dimY + 2) * k;
					if (grid[index] > maxHeat) maxHeat = grid[index];
					if (grid[index] < minHeat) minHeat = grid[index];
				}
			}
		}
		std::cout << "Max: " << maxHeat << std::endl;
		std::cout << "Min: " << minHeat << std::endl;
	}

	delete[] grid;
	delete[] next;

	timer.Stop("TOTAL");

	double totalTime = timer.GetCustomTime("TOTAL");
	double computeTime = timer.GetCustomTime("COMPUTE");
	std::cout << "Total: " << totalTime << std::endl;
	std::cout << "Compute: " << computeTime << std::endl;
	std::cout << "GFLOPS: " << 12 * timer.CalculateBLUPS("COMPUTE", dimX * dimY * dimZ, iteration) << std::endl;
}


void InitialData(int dimX, int dimY, int dimZ, double* data)
{
	double meanX = (dimX - 1) / 2.0;
	double meanY = (dimY - 1) / 2.0;
	double meanZ = (dimZ - 1) / 2.0;

	double sumX = 0.0;
	double sumY = 0.0;
	double sumZ = 0.0;

	for (int i = 0; i < dimX; i++)
	{
		sumX += pow(i - meanX, 2);
	}

	for (int j = 0; j < dimY; j++)
	{
		sumY += pow(j - meanY, 2);
	}

	for (int k = 0; k < dimZ; k++)
	{
		sumZ += pow(k - meanZ, 2);
	}

	double varX = sumX / (dimX - 1);
	double varY = sumY / (dimY - 1);
	double varZ = sumZ / (dimZ - 1);

	for (int k = 0; k < dimZ; k++)
	{
		for (int j = 0; j < dimY; j++)
		{
			for (int i = 0; i < dimX; i++)
			{
				double val = exp(-(pow(i - meanX, 2) / (2 * varX) + pow(j - meanY, 2) / (2 * varY) + pow(k - meanZ, 2) / (2 * varZ)));
				data[(i + 1) + (dimX + 2) * (j + 1) + (dimX + 2) * (dimY + 2) * (k + 1)] = val;
			}
		}
	}
}


void Simulate(int dimX, int dimY, int dimZ, double cx, double cy, double cz, double* cur, double* next)
{
	for (int k = 1; k <= dimZ; k++)
	{
		for (int j = 1; j <= dimY; j++)
		{
			for (int i = 1; i <= dimX; i++)
			{
				double val = cur[Index1D(i, j, k)] 
					+ cx * (cur[Index1D(i - 1, j, k)] + cur[Index1D(i + 1, j, k)] - 2 * cur[Index1D(i, j, k)])
					+ cy * (cur[Index1D(i, j - 1, k)] + cur[Index1D(i, j + 1, k)] - 2 * cur[Index1D(i, j, k)])
					+ cz * (cur[Index1D(i, j, k - 1)] + cur[Index1D(i, j, k + 1)] - 2 * cur[Index1D(i, j, k)]);

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