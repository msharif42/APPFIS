/**
 *
 * Cuda version of Heat Diffusion
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef HEAT_CUDA_HPP
#define HEAT_CUDA_HPP


#include <iostream>
#include <limits>
#include "../../Appfis/Timer.hpp"
#include "../../Appfis/Utils.hpp"
using namespace std;


void HeatDiffusion(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int dimZ, double* data);
__global__ void Simulate(int dimX, int dimY, int dimZ, double cx, double cy, double cz, double* cur, double* next);
__device__ int Index1D(int i, int j, int k);
__device__ int strideX;
__device__ int strideY;


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

	int h_strideX = (dimX + 2);
	int h_strideY = (dimY + 2);
	cudaMemcpyToSymbol(strideX, &h_strideX, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(strideY, &h_strideY, sizeof(int), 0, cudaMemcpyHostToDevice);

	double* h_grid = new double[(dimX + 2) * (dimY + 2) * (dimZ + 2)]();
	double* d_grid;
	double* d_next;
	double* d_tmp;

	size_t bytes = sizeof(double) * (dimX + 2) * (dimY + 2) * (dimZ + 2);
	cudaMalloc(&d_grid, bytes);
	cudaMalloc(&d_next, bytes);

	InitialData(dimX, dimY, dimZ, h_grid);

	cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice);
	dim3 blockSize(8, 8, 4);
	int bx = (int)ceil(dimX / (float)8);
	int by = (int)ceil(dimY / (float)8);
	int bz = (int)ceil(dimZ / (float)4);
	dim3 gridSize(bx, by, bz);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		Simulate << <gridSize, blockSize >> > (dimX, dimY, dimZ, cx, cy, cz, d_grid, d_next);
		cudaDeviceSynchronize();

		d_tmp = d_grid;
		d_grid = d_next;
		d_next = d_tmp;
	}
	timer.Stop("COMPUTE");

	cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost);

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
					if (h_grid[index] > maxHeat) maxHeat = h_grid[index];
					if (h_grid[index] < minHeat) minHeat = h_grid[index];
				}
			}
		}
		std::cout << "Max: " << maxHeat << std::endl;
		std::cout << "Min: " << minHeat << std::endl;
	}

	cudaFree(d_grid);
	cudaFree(d_next);
	delete[] h_grid;

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


__global__ void Simulate(int dimX, int dimY, int dimZ, double cx, double cy, double cz, double* cur, double* next)
{
	int k = blockDim.z * blockIdx.z + threadIdx.z + 1;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (k <= dimZ && j <= dimY && i <= dimX) 
	{
		double val = cur[Index1D(i, j, k)]
			+ cx * (cur[Index1D(i - 1, j, k)] + cur[Index1D(i + 1, j, k)] - 2 * cur[Index1D(i, j, k)])
			+ cy * (cur[Index1D(i, j - 1, k)] + cur[Index1D(i, j + 1, k)] - 2 * cur[Index1D(i, j, k)])
			+ cz * (cur[Index1D(i, j, k - 1)] + cur[Index1D(i, j, k + 1)] - 2 * cur[Index1D(i, j, k)]);

		next[Index1D(i, j, k)] = val;
	}
}


__device__ int Index1D(int i, int j, int k)
{
	return i + j * strideX + k * strideX * strideY;
}


#endif