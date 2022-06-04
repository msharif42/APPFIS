/**
 *
 * Cuda version of FDTD3d
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef FDTD3D_CUDA_HPP
#define FDTD3D_CUDA_HPP


#include <iostream>
#include <limits>
#include "../../Appfis/Timer.hpp"
#include "../../Appfis/Utils.hpp"
using namespace std;


void FDTD3d(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int dimZ, int radius, float* data);
__global__ void Simulate(int dimX, int dimY, int dimZ, float coeff, int radius, float* cur, float* next);
__device__ int Index1D(int i, int j, int k);
__device__ int strideX;
__device__ int strideY;


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

	int h_strideX = (dimX + 2);
	int h_strideY = (dimY + 2);
	cudaMemcpyToSymbol(strideX, &h_strideX, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(strideY, &h_strideY, sizeof(int), 0, cudaMemcpyHostToDevice);

	float* h_grid = new float[(dimX + 2 * radius) * (dimY + 2 * radius) * (dimZ + 2 * radius)]();
	float* d_grid;
	float* d_next;
	float* d_tmp;

	size_t bytes = sizeof(float) * (dimX + 2 * radius) * (dimY + 2 * radius) * (dimZ + 2 * radius);
	cudaMalloc(&d_grid, bytes);
	cudaMalloc(&d_next, bytes);

	InitialData(dimX, dimY, dimZ, radius, h_grid);

	cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice);
	dim3 blockSize(8, 8, 4);
	int bx = (int)ceil(dimX / (float)8);
	int by = (int)ceil(dimY / (float)8);
	int bz = (int)ceil(dimZ / (float)4);
	dim3 gridSize(bx, by, bz);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		Simulate << <gridSize, blockSize >> > (dimX, dimY, dimZ, coeff, radius, d_grid, d_next);
		cudaDeviceSynchronize();

		d_tmp = d_grid;
		d_grid = d_next;
		d_next = d_tmp;
	}
	timer.Stop("COMPUTE");

	cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_grid);
	cudaFree(d_next);
	delete[] h_grid;

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


__global__ void Simulate(int dimX, int dimY, int dimZ, float coeff, int radius, float* cur, float* next)
{
	int k = blockDim.z * blockIdx.z + threadIdx.z + radius;
	int j = blockDim.y * blockIdx.y + threadIdx.y + radius;
	int i = blockDim.x * blockIdx.x + threadIdx.x + radius;

	if (k < dimZ + radius && j < dimY + radius && i < dimX + radius) {
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


__device__ int Index1D(int i, int j, int k)
{
	return i + j * strideX + k * strideX * strideY;
}


#endif