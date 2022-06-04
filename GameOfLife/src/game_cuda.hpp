/**
 *
 * Cuda version of Game Of Life
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef GAME_CUDA_HPP
#define GAME_CUDA_HPP


#include <iostream>
#include "../../Appfis/Timer.hpp"
#include "../../Appfis/Utils.hpp"
using namespace std;


void GameOfLife(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int* data);
__global__ void GhostRows(int dimX, int dimY, int* cur);
__global__ void GhostCols(int dimX, int dimY, int* cur);
__global__ void Simulate(int dimX, int dimY, int* cur, int* next);
__device__ int Index1D(int i, int j);
__device__ int stride;


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

	int h_stride = (dimX + 2);
	cudaMemcpyToSymbol(stride, &h_stride, sizeof(int), 0, cudaMemcpyHostToDevice);

	int* h_grid = new int[(dimX + 2) * (dimY + 2)]();
	int* d_grid;
	int* d_next;
	int* d_tmp;

	size_t bytes = sizeof(int) * (dimX + 2) * (dimY + 2);
	cudaMalloc(&d_grid, bytes);
	cudaMalloc(&d_next, bytes);

	InitialData(dimX, dimY, h_grid);

	cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice);
	dim3 blockSize(16, 16, 1);
	int bx = (int)ceil(dimX / (float)16);
	int by = (int)ceil(dimY / (float)16);
	dim3 gridSize(bx, by, 1);

	dim3 cpyBlockSize(16, 1, 1);
	dim3 cpyGridRowsGridSize((int)ceil(dimX / (float)cpyBlockSize.x), 1, 1);
	dim3 cpyGridColsGridSize((int)ceil((dimY + 2) / (float)cpyBlockSize.x), 1, 1);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		GhostRows << <cpyGridRowsGridSize, cpyBlockSize >> > (dimX, dimY, d_grid);
		GhostCols << <cpyGridColsGridSize, cpyBlockSize >> > (dimX, dimY, d_grid);
		Simulate << <gridSize, blockSize >> > (dimX, dimY, d_grid, d_next);
		cudaDeviceSynchronize();

		d_tmp = d_grid;
		d_grid = d_next;
		d_next = d_tmp;
	}
	timer.Stop("COMPUTE");

	cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost);

	if (outputFlag)
	{
		int liveCount = 0;
		for (int j = 1; j <= dimY; j++)
		{
			for (int i = 1; i <= dimX; i++)
			{
				liveCount += h_grid[i + j * (dimX + 2)];
			}
		}
		std::cout << "Live Count: " << liveCount << std::endl;
	}

	cudaFree(d_grid);
	cudaFree(d_next);
	delete[] h_grid;

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


__global__ void GhostRows(int dimX, int dimY, int* cur)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (i <= dimX)
	{
		cur[Index1D(i, 0)] = cur[Index1D(i, dimY)];
		cur[Index1D(i, dimY + 1)] = cur[Index1D(i, 1)];
	}
}


__global__ void GhostCols(int dimX, int dimY, int* cur)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j <= dimY + 1)
	{
		cur[Index1D(0, j)] = cur[Index1D(dimX, j)];
		cur[Index1D(dimX + 1, j)] = cur[Index1D(1, j)];
	}
}


__global__ void Simulate(int dimX, int dimY, int* cur, int* next)
{
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (j <= dimY && i <= dimX) 
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


__device__ int Index1D(int i, int j)
{
	return i + j * stride;
}


#endif