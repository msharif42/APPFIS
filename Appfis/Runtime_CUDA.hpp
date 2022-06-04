/**
 *
 * This is the file containing methods that are useful in cuda runtime
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef RUNTIME_CUDA_HPP
#define RUNTIME_CUDA_HPP


#ifdef APPFIS_CUDA


#include "Common.hpp"
#include "Array.hpp"
#include "HaloBuffer.hpp"
#include "Grid.hpp"
#include "Communication.hpp"
#include "Config.hpp"
#include <utility>


namespace APPFIS
{
	template<typename T>
	void HaloExchangeGPU(Grid<T>* grid);

	template<typename T, typename... TArgs>
	void Execute1DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void Execute2DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void Execute3DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteSerial1DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteSerial2DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteSerial3DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteOverlap1DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteOverlap2DGPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteOverlap3DGPU(Config<T>& config, TArgs... args);

	template<typename T>
	void Execute1DReduceGPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void Execute2DReduceGPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void Execute3DReduceGPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void CopyArrayToBufferGPUHelper(Grid<T>* grid, int face);

	template<typename T>
	void CopyBufferToArrayGPUHelper(Grid<T>* grid, int face);

	template<typename T>
	void ExchangeHelperGPUToCPU(Config<T>& config);

	template<typename T>
	void ExchangeHelperGPUToCPU2(Grid<T>* grid);

	template<typename T>
	void ExchangeHelperGPUToCPU3(Grid<T>* grid);

	template<typename T>
	void ExchangeHelperGPU(Config<T>& config);

	template<typename T>
	void ExchangeHelperCPUToGPU(Config<T>& config);

	template<typename T>
	void ExchangeHelperCPUToGPU2(Grid<T>* grid);

	template<typename T>
	void ExchangeHelperCPUToGPU3(Grid<T>* grid);


	template<typename... TArgs>
	__global__ void Kernel1D(int start, int end, TArgs... args);

	template<typename... TArgs>
	__global__ void Kernel2D(int start1, int end1, int start2, int end2, TArgs... args);

	template<typename... TArgs>
	__global__ void Kernel3D(int start1, int end1, int start2, int end2, int start3, int end3, TArgs... args);

	template<typename T>
	__global__ void Kernel1DReduceSum(int start, int end, T* arr_i, T* arr_o);

	template<typename T>
	__global__ void Kernel2DReduceSum(int start1, int end1, int start2, int end2, int dimX, T* arr_i, T* arr_o);

	template<typename T>
	__global__ void Kernel3DReduceSum(int start1, int end1, int start2, int end2, int start3, int end3, int dimX, int dimY, T* arr_i, T* arr_o);

	template<typename T>
	__global__ void Kernel1DReduceMax(int start, int end, T* arr_i, T* arr_o, T min);

	template<typename T>
	__global__ void Kernel2DReduceMax(int start1, int end1, int start2, int end2, int dimX, T* arr_i, T* arr_o, T min);

	template<typename T>
	__global__ void Kernel3DReduceMax(int start1, int end1, int start2, int end2, int start3, int end3, int dimX, int dimY, T* arr_i, T* arr_o, T min);

	template<typename T>
	__global__ void Kernel1DReduceMin(int start, int end, T* arr_i, T* arr_o, T max);

	template<typename T>
	__global__ void Kernel2DReduceMin(int start1, int end1, int start2, int end2, int dimX, T* arr_i, T* arr_o, T max);

	template<typename T>
	__global__ void Kernel3DReduceMin(int start1, int end1, int start2, int end2, int start3, int end3, int dimX, int dimY, T* arr_i, T* arr_o, T max);

	template<typename T>
	__global__ void KernelReduceSumLoop(int size, T* arr);

	template<typename T>
	__global__ void KernelReduceMaxLoop(int size, T* arr, T min);

	template<typename T>
	__global__ void KernelReduceMinLoop(int size, T* arr, T max);

	template<typename T>
	__global__ void ArrayToBuffer1D_LEFT(Array<T>* arr, T* buffer, int dimX, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer1D_RIGHT(Array<T>* arr, T* buffer, int dimX, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer2D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer2D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer2D_Y_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer2D_Y_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer3D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer3D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer3D_Y_FRONT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer3D_Y_BACK(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer3D_Z_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void ArrayToBuffer3D_Z_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray1D_LEFT(Array<T>* arr, T* buffer, int dimX, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray1D_RIGHT(Array<T>* arr, T* buffer, int dimX, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray2D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray2D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray2D_Y_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray2D_Y_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray3D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray3D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray3D_Y_FRONT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray3D_Y_BACK(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray3D_Z_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);

	template<typename T>
	__global__ void BufferToArray3D_Z_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer);


	template<typename T>
	void HaloExchangeGPU(Grid<T>* grid)
	{
		ExchangeHelperGPUToCPU2(grid);
		ExchangeHelperGPUToCPU3(grid);
		cudaStreamSynchronize(GetCudaStream());

		Array<T>* arr = grid->GetSubGridCPU();
		HaloBuffer<T>* buffer = grid->GetBuffer();
		ExchangeGPU(arr, buffer);

		ExchangeHelperCPUToGPU2(grid);
		ExchangeHelperCPUToGPU3(grid);
		//cudaStreamSynchronize(GetCudaStream());

		grid->Swap();
	}


	template<typename T, typename... TArgs>
	void Execute1DGPU(Config<T>& config, TArgs... args)
	{
		bool overlap = IsOverlap();

		if (overlap && config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExecuteOverlap1DGPU(config, args...);
		}
		else
		{
			ExecuteSerial1DGPU(config, args...);
		}
	}


	template<typename T, typename... TArgs>
	void Execute2DGPU(Config<T>& config, TArgs... args)
	{
		bool overlap = IsOverlap();

		if (overlap && config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExecuteOverlap2DGPU(config, args...);
		}
		else
		{
			ExecuteSerial2DGPU(config, args...);
		}
	}


	template<typename T, typename... TArgs>
	void Execute3DGPU(Config<T>& config, TArgs... args)
	{
		bool overlap = IsOverlap();

		if (overlap && config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExecuteOverlap3DGPU(config, args...);
		}
		else
		{
			ExecuteSerial3DGPU(config, args...);
		}
	}


	template<typename T, typename... TArgs>
	void ExecuteSerial1DGPU(Config<T>& config, TArgs... args)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();

		dim3 dimBlock(256);
		int bx = (endX - startX + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid(bx);

		Kernel1D << < dimGrid, dimBlock, 0, GetCudaStream() >> > (startX, endX, args...);

		if (config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExchangeHelperGPUToCPU(config);
			cudaStreamSynchronize(GetCudaStream());
			ExchangeHelperGPU(config);
			ExchangeHelperCPUToGPU(config);
		}

		//cudaStreamSynchronize(GetCudaStream());
		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteOverlap1DGPU(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();
		int dimX = config.GetDimX();
		int startX = config.GetStartX();
		int endX = config.GetEndX();

		int endTempX = 2 * ghostLayer;
		int startTempX = dimX - 2 * ghostLayer;

		dim3 dimBlock(1);
		int bx = (endTempX - startX + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid(bx);

		int bx2 = (endX - startTempX + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid2(bx2);

		Kernel1D << < dimGrid, dimBlock, 0, GetCudaStream() >> > (startX, endTempX, args...);
		Kernel1D << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (startTempX, endX, args...);

		ExchangeHelperGPUToCPU(config);
		cudaStreamSynchronize(GetCudaStream());


		startX = 2 * ghostLayer;
		endX = dimX - 2 * ghostLayer;

		dim3 dimBlock3(256);
		int bx3 = (endX - startX + dimBlock3.x - 1) / dimBlock3.x;
		dim3 dimGrid3(bx3);

		Kernel1D << < dimGrid3, dimBlock3, 0, GetCudaStream() >> > (startX, endX, args...);

		ExchangeHelperGPU(config);
		ExchangeHelperCPUToGPU(config);
		//cudaStreamSynchronize(GetCudaStream());

		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteSerial2DGPU(Config<T>& config, TArgs... args)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();

		dim3 dimBlock(16, 16);
		int bx = (endX - startX + dimBlock.x - 1) / dimBlock.x;
		int by = (endY - startY + dimBlock.y - 1) / dimBlock.y;
		dim3 dimGrid(bx, by);

		Kernel2D << < dimGrid, dimBlock, 0, GetCudaStream() >> > (startX, endX, startY, endY, args...);

		if (config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExchangeHelperGPUToCPU(config);
			cudaStreamSynchronize(GetCudaStream());
			ExchangeHelperGPU(config);
			ExchangeHelperCPUToGPU(config);
		}
		//cudaStreamSynchronize(GetCudaStream());

		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteOverlap2DGPU(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();
		int dimX = config.GetDimX();
		int dimY = config.GetDimY();
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();

		int startTempX = dimX - 2 * ghostLayer;
		int startTempY = dimY - 2 * ghostLayer;
		int endTempX = 2 * ghostLayer;
		int endTempY = 2 * ghostLayer;

		dim3 dimBlock(1, 256);
		int bx = (2 * ghostLayer - startX + dimBlock.x - 1) / dimBlock.x;
		int by = (endY - startY + dimBlock.y - 1) / dimBlock.y;
		dim3 dimGrid(bx, by);

		int bx_ = (endX - dimX + 2 * ghostLayer + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid_(bx_, by);

		Kernel2D << < dimGrid, dimBlock, 0, GetCudaStream() >> > (startX, endTempX, startY, endY, args...);
		Kernel2D << < dimGrid_, dimBlock, 0, GetCudaStream() >> > (startTempX, endX, startY, endY, args...);

		dim3 dimBlock2(256, 1);
		int bx2 = (dimX - 4 * ghostLayer + dimBlock2.x - 1) / dimBlock2.x;
		int by2 = (2 * ghostLayer - startY + dimBlock2.y - 1) / dimBlock2.y;
		dim3 dimGrid2(bx2, by2);

		int by2_ = (endY - dimY + 2 * ghostLayer + dimBlock2.y - 1) / dimBlock2.y;
		dim3 dimGrid2_(bx2, by2_);

		Kernel2D << < dimGrid2, dimBlock2, 0, GetCudaStream() >> > (endTempX, startTempX, startY, endTempY, args...);
		Kernel2D << < dimGrid2_, dimBlock2, 0, GetCudaStream() >> > (endTempX, startTempX, startTempY, endY, args...);

		ExchangeHelperGPUToCPU(config);
		cudaStreamSynchronize(GetCudaStream());


		startX = 2 * ghostLayer;
		startY = 2 * ghostLayer;
		endX = dimX - 2 * ghostLayer;
		endY = dimY - 2 * ghostLayer;

		dim3 dimBlock3(16, 16);
		int bx3 = (endX - startX + dimBlock3.x - 1) / dimBlock3.x;
		int by3 = (endY - startY + dimBlock3.y - 1) / dimBlock3.y;
		dim3 dimGrid3(bx3, by3);

		Kernel2D << < dimGrid3, dimBlock3, 0, GetCudaStream() >> > (startX, endX, startY, endY, args...);

		ExchangeHelperGPU(config);
		ExchangeHelperCPUToGPU(config);
		//cudaStreamSynchronize(GetCudaStream());

		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteSerial3DGPU(Config<T>& config, TArgs... args)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();
		int startZ = config.GetStartZ();
		int endZ = config.GetEndZ();

		dim3 dimBlock(8, 8, 4);
		int bx = (endX - startX + dimBlock.x - 1) / dimBlock.x;
		int by = (endY - startY + dimBlock.y - 1) / dimBlock.y;
		int bz = (endZ - startZ + dimBlock.z - 1) / dimBlock.z;
		dim3 dimGrid(bx, by, bz);

		Kernel3D << < dimGrid, dimBlock, 0, GetCudaStream() >> > (startX, endX, startY, endY, startZ, endZ, args...);

		if (config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExchangeHelperGPUToCPU(config);
			cudaStreamSynchronize(GetCudaStream());
			ExchangeHelperGPU(config);
			ExchangeHelperCPUToGPU(config);
		}
		//cudaStreamSynchronize(GetCudaStream());

		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteOverlap3DGPU(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();
		int dimX = config.GetDimX();
		int dimY = config.GetDimY();
		int dimZ = config.GetDimZ();
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();
		int startZ = config.GetStartZ();
		int endZ = config.GetEndZ();

		int startTempX = dimX - 2 * ghostLayer;
		int startTempY = dimY - 2 * ghostLayer;
		int startTempZ = dimZ - 2 * ghostLayer;

		int endTempX = 2 * ghostLayer;
		int endTempY = 2 * ghostLayer;
		int endTempZ = 2 * ghostLayer;

		dim3 dimBlock(1, 16, 16);
		int bx = (2 * ghostLayer - startX + dimBlock.x - 1) / dimBlock.x;
		int by = (endY - startY + dimBlock.y - 1) / dimBlock.y;
		int bz = (endZ - startZ + dimBlock.z - 1) / dimBlock.z;
		dim3 dimGrid(bx, by, bz);

		int bx_ = (endX - dimX + 2 * ghostLayer + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid_(bx_, by, bz);

		Kernel3D << < dimGrid, dimBlock, 0, GetCudaStream() >> > (startX, endTempX, startY, endY, startZ, endZ, args...);
		Kernel3D << < dimGrid_, dimBlock, 0, GetCudaStream() >> > (startTempX, endX, startY, endY, startZ, endZ, args...);

		dim3 dimBlock2(16, 1, 16);
		int bx2 = (dimX - 4 * ghostLayer + dimBlock2.x - 1) / dimBlock2.x;
		int by2 = (2 * ghostLayer - startY + dimBlock2.y - 1) / dimBlock2.y;
		int bz2 = (endZ - startZ + dimBlock2.z - 1) / dimBlock2.z;

		dim3 dimGrid2(bx2, by2, bz2);

		int by2_ = (endY - dimY + 2 * ghostLayer + dimBlock2.y - 1) / dimBlock2.y;
		dim3 dimGrid2_(bx2, by2_, bz2);

		Kernel3D << < dimGrid2, dimBlock2, 0, GetCudaStream() >> > (endTempX, startTempX, startY, endTempY, startZ, endZ, args...);
		Kernel3D << < dimGrid2_, dimBlock2, 0, GetCudaStream() >> > (endTempX, startTempX, startTempY, endY, startZ, endZ, args...);

		dim3 dimBlock3(16, 16, 1);
		int bx3 = (dimX - 4 * ghostLayer + dimBlock3.x - 1) / dimBlock3.x;
		int by3 = (dimY - 4 * ghostLayer + dimBlock3.y - 1) / dimBlock3.y;
		int bz3 = (2 * ghostLayer - startZ + dimBlock3.z - 1) / dimBlock3.z;

		dim3 dimGrid3(bx3, by3, bz3);

		int bz3_ = (endZ - dimZ + 2 * ghostLayer + dimBlock3.z - 1) / dimBlock3.z;
		dim3 dimGrid3_(bx3, by3, bz3_);

		Kernel3D << < dimGrid3, dimBlock3, 0, GetCudaStream() >> > (endTempX, startTempX, endTempY, startTempY, startZ, endTempZ, args...);
		Kernel3D << < dimGrid3_, dimBlock3, 0, GetCudaStream() >> > (endTempX, startTempX, endTempY, startTempY, startTempZ, endZ, args...);

		ExchangeHelperGPUToCPU(config);
		cudaStreamSynchronize(GetCudaStream());

		startX = 2 * ghostLayer;
		startY = 2 * ghostLayer;
		startZ = 2 * ghostLayer;

		endX = dimX - 2 * ghostLayer;
		endY = dimY - 2 * ghostLayer;
		endZ = dimZ - 2 * ghostLayer;

		dim3 dimBlock4(8, 8, 4);
		int bx4 = (endX - startX + dimBlock4.x - 1) / dimBlock4.x;
		int by4 = (endY - startY + dimBlock4.y - 1) / dimBlock4.y;
		int bz4 = (endZ - startZ + dimBlock4.z - 1) / dimBlock4.z;
		dim3 dimGrid4(bx4, by4, bz4);

		Kernel3D << < dimGrid4, dimBlock4, 0, GetCudaStream() >> > (startX, endX, startY, endY, startZ, endZ, args...);

		ExchangeHelperGPU(config);
		ExchangeHelperCPUToGPU(config);
		//cudaStreamSynchronize(GetCudaStream());

		SwapHelper(config);
	}


	template<typename T>
	void Execute1DReduceGPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
		int dimX = config.GetDimX();
		int startX = config.GetStartX();
		int endX = config.GetEndX();

		T* arr_i = grid.GetSubGridsGPUData();
		T* arr_o = grid.GetSubGridsNextGPUData();

		dim3 dimBlock(256);
		int bx = (dimX + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid(bx);

		T local;

		if (rType == SUM)
		{
			Kernel1DReduceSum << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, arr_i, arr_o);
		}
		else if (rType == MAX)
		{
			T min_val = (std::numeric_limits<T>::min)();
			Kernel1DReduceMax << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, arr_i, arr_o, min_val);
		}
		else if (rType == MIN)
		{
			T max_val = (std::numeric_limits<T>::max)();
			Kernel1DReduceMin << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, arr_i, arr_o, max_val);
		}

		while (bx > 1)
		{
			int size = bx;
			bx = (bx + dimBlock.x - 1) / dimBlock.x;
			dim3 dimGrid2(bx);

			if (rType == SUM)
			{
				KernelReduceSumLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o);
			}
			else if (rType == MAX)
			{
				T min_val = (std::numeric_limits<T>::min)();
				KernelReduceMaxLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o, min_val);
			}
			else if (rType == MIN)
			{
				T max_val = (std::numeric_limits<T>::max)();
				KernelReduceMinLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o, max_val);
			}
		}

		cudaMemcpyAsync(&local, arr_o, sizeof(T), cudaMemcpyDeviceToHost, GetCudaStream());
		cudaStreamSynchronize(GetCudaStream());

		var = AllReduce(local, rType);
	}


	template<typename T>
	void Execute2DReduceGPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
		int dimX = config.GetDimX();
		int dimY = config.GetDimY();
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();

		T* arr_i = grid.GetSubGridsGPUData();
		T* arr_o = grid.GetSubGridsNextGPUData();

		dim3 dimBlock(256);
		int bx = (dimX * dimY + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid(bx);

		T local;

		if (rType == SUM)
		{
			Kernel2DReduceSum << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, startY, endY, dimX, arr_i, arr_o);
		}
		else if (rType == MAX)
		{
			T min_val = (std::numeric_limits<T>::min)();
			Kernel2DReduceMax << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, startY, endY, dimX, arr_i, arr_o, min_val);
		}
		else if (rType == MIN)
		{
			T max_val = (std::numeric_limits<T>::max)();
			Kernel2DReduceMin << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, startY, endY, dimX, arr_i, arr_o, max_val);
		}

		while (bx > 1)
		{
			int size = bx;
			bx = (bx + dimBlock.x - 1) / dimBlock.x;
			dim3 dimGrid2(bx);

			if (rType == SUM)
			{
				KernelReduceSumLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o);
			}
			else if (rType == MAX)
			{
				T min_val = (std::numeric_limits<T>::min)();
				KernelReduceMaxLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o, min_val);
			}
			else if (rType == MIN)
			{
				T max_val = (std::numeric_limits<T>::max)();
				KernelReduceMinLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o, max_val);
			}
		}

		cudaMemcpyAsync(&local, arr_o, sizeof(T), cudaMemcpyDeviceToHost, GetCudaStream());
		cudaStreamSynchronize(GetCudaStream());

		var = AllReduce(local, rType);
	}


	template<typename T>
	void Execute3DReduceGPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
		int dimX = config.GetDimX();
		int dimY = config.GetDimY();
		int dimZ = config.GetDimZ();
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();
		int startZ = config.GetStartZ();
		int endZ = config.GetEndZ();

		T* arr_i = grid.GetSubGridsGPUData();
		T* arr_o = grid.GetSubGridsNextGPUData();

		dim3 dimBlock(256);
		int bx = (dimX * dimY * dimZ + dimBlock.x - 1) / dimBlock.x;
		dim3 dimGrid(bx);

		T local;

		if (rType == SUM)
		{
			Kernel3DReduceSum << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, startY, endY, startZ, endZ, dimX, dimY, arr_i, arr_o);
		}
		else if (rType == MAX)
		{
			T min_val = (std::numeric_limits<T>::min)();
			Kernel3DReduceMax << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, startY, endY, startZ, endZ, dimX, dimY, arr_i, arr_o, min_val);
		}
		else if (rType == MIN)
		{
			T max_val = (std::numeric_limits<T>::max)();
			Kernel3DReduceMin << < dimGrid, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (startX, endX, startY, endY, startZ, endZ, dimX, dimY, arr_i, arr_o, max_val);
		}

		while (bx > 1)
		{
			int size = bx;
			bx = (bx + dimBlock.x - 1) / dimBlock.x;
			dim3 dimGrid2(bx);

			if (rType == SUM)
			{
				KernelReduceSumLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o);
			}
			else if (rType == MAX)
			{
				T min_val = (std::numeric_limits<T>::min)();
				KernelReduceMaxLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o, min_val);
			}
			else if (rType == MIN)
			{
				T max_val = (std::numeric_limits<T>::max)();
				KernelReduceMinLoop << < dimGrid2, dimBlock, dimBlock.x * sizeof(T), GetCudaStream() >> > (size, arr_o, max_val);
			}
		}

		cudaMemcpyAsync(&local, arr_o, sizeof(T), cudaMemcpyDeviceToHost, GetCudaStream());
		cudaStreamSynchronize(GetCudaStream());

		var = AllReduce(local, rType);
	}


	template<typename T>
	void CopyArrayToBufferGPUHelper(Grid<T>* grid, int face)
	{
		Array<T>* harr = grid->GetSubGridCPU();
		Array<T>* darr = grid->GetSubGridNext();

		int ghostLayer = GetGhostLayer();
		int dim = harr->GetDimCount();

		if (dim == DIM_1D)
		{
			int dimX = harr->GetDim(X);

			dim3 dimBlock(1);
			int bx = (ghostLayer + dimBlock.x - 1) / dimBlock.x;
			dim3 dimGrid(bx);

			if (face == LEFT)
			{
				ArrayToBuffer1D_LEFT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(LEFT), dimX, ghostLayer);
			}
			else if (face == RIGHT)
			{
				ArrayToBuffer1D_RIGHT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(RIGHT), dimX, ghostLayer);
			}
		}
		else if (dim == DIM_2D)
		{
			int dimX = harr->GetDim(X);
			int dimY = harr->GetDim(Y);

			dim3 dimBlock(1, 256);
			int bx = (ghostLayer + dimBlock.x - 1) / dimBlock.x;
			int by = (dimY + dimBlock.y - 1) / dimBlock.y;
			int by2 = (dimX + dimBlock.y - 1) / dimBlock.y;

			dim3 dimGrid(bx, by);
			dim3 dimGrid2(bx, by2);

			if (face == LEFT)
			{
				ArrayToBuffer2D_X_LEFT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(LEFT), dimX, dimY, ghostLayer);
			}
			else if (face == RIGHT)
			{
				ArrayToBuffer2D_X_RIGHT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(RIGHT), dimX, dimY, ghostLayer);
			}
			else if (face == UP)
			{
				ArrayToBuffer2D_Y_UP << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(UP), dimX, dimY, ghostLayer);
			}
			else if (face == DOWN)
			{
				ArrayToBuffer2D_Y_DOWN << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(DOWN), dimX, dimY, ghostLayer);
			}
		}
		else if (dim == DIM_3D)
		{
			int dimX = harr->GetDim(X);
			int dimY = harr->GetDim(Y);
			int dimZ = harr->GetDim(Z);

			dim3 dimBlock(1, 16, 16);
			int bx = (ghostLayer + dimBlock.x - 1) / dimBlock.x;
			int by = (dimX + dimBlock.y - 1) / dimBlock.y;
			int by2 = (dimY + dimBlock.y - 1) / dimBlock.y;
			int by3 = (dimZ + dimBlock.y - 1) / dimBlock.y;

			dim3 dimGrid(bx, by2, by3);
			dim3 dimGrid2(bx, by, by3);
			dim3 dimGrid3(bx, by, by2);

			if (face == LEFT)
			{
				ArrayToBuffer3D_X_LEFT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(LEFT), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == RIGHT)
			{
				ArrayToBuffer3D_X_RIGHT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(RIGHT), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == FRONT)
			{
				ArrayToBuffer3D_Y_FRONT << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(FRONT), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == BACK)
			{
				ArrayToBuffer3D_Y_BACK << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(BACK), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == DOWN)
			{
				ArrayToBuffer3D_Z_DOWN << < dimGrid3, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(DOWN), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == UP)
			{
				ArrayToBuffer3D_Z_UP << < dimGrid3, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetSendBufferGPU(UP), dimX, dimY, dimZ, ghostLayer);
			}
		}
	}


	template<typename T>
	void CopyBufferToArrayGPUHelper(Grid<T>* grid, int face)
	{
		Array<T>* harr = grid->GetSubGridCPU();
		Array<T>* darr = grid->GetSubGridNext();

		int ghostLayer = GetGhostLayer();
		int dim = harr->GetDimCount();

		if (dim == DIM_1D)
		{
			int dimX = harr->GetDim(X);

			dim3 dimBlock(1);
			int bx = (ghostLayer + dimBlock.x - 1) / dimBlock.x;
			dim3 dimGrid(bx);

			if (face == LEFT)
			{
				BufferToArray1D_LEFT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(LEFT), dimX, ghostLayer);
			}
			else if (face == RIGHT)
			{
				BufferToArray1D_RIGHT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(RIGHT), dimX, ghostLayer);
			}
		}
		else if (dim == DIM_2D)
		{
			int dimX = harr->GetDim(X);
			int dimY = harr->GetDim(Y);

			dim3 dimBlock(1, 256);
			int bx = (ghostLayer + dimBlock.x - 1) / dimBlock.x;
			int by = (dimY + dimBlock.y - 1) / dimBlock.y;
			int by2 = (dimX + dimBlock.y - 1) / dimBlock.y;

			dim3 dimGrid(bx, by);
			dim3 dimGrid2(bx, by2);

			if (face == LEFT)
			{
				BufferToArray2D_X_LEFT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(LEFT), dimX, dimY, ghostLayer);
			}
			else if (face == RIGHT)
			{
				BufferToArray2D_X_RIGHT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(RIGHT), dimX, dimY, ghostLayer);
			}
			else if (face == UP)
			{
				BufferToArray2D_Y_UP << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(UP), dimX, dimY, ghostLayer);
			}
			else if (face == DOWN)
			{
				BufferToArray2D_Y_DOWN << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(DOWN), dimX, dimY, ghostLayer);
			}
		}
		else if (dim == DIM_3D)
		{
			int dimX = harr->GetDim(X);
			int dimY = harr->GetDim(Y);
			int dimZ = harr->GetDim(Z);

			dim3 dimBlock(1, 16, 16);
			int bx = (ghostLayer + dimBlock.x - 1) / dimBlock.x;
			int by = (dimX + dimBlock.y - 1) / dimBlock.y;
			int by2 = (dimY + dimBlock.y - 1) / dimBlock.y;
			int by3 = (dimZ + dimBlock.y - 1) / dimBlock.y;

			dim3 dimGrid(bx, by2, by3);
			dim3 dimGrid2(bx, by, by3);
			dim3 dimGrid3(bx, by, by2);

			if (face == LEFT)
			{
				BufferToArray3D_X_LEFT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(LEFT), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == RIGHT)
			{
				BufferToArray3D_X_RIGHT << < dimGrid, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(RIGHT), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == FRONT)
			{
				BufferToArray3D_Y_FRONT << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(FRONT), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == BACK)
			{
				BufferToArray3D_Y_BACK << < dimGrid2, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(BACK), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == DOWN)
			{
				BufferToArray3D_Z_DOWN << < dimGrid3, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(DOWN), dimX, dimY, dimZ, ghostLayer);
			}
			else if (face == UP)
			{
				BufferToArray3D_Z_UP << < dimGrid3, dimBlock, 0, GetCudaStream() >> > (darr, grid->GetRecvBufferGPU(UP), dimX, dimY, dimZ, ghostLayer);
			}
		}
	}


	template<typename T>
	void ExchangeHelperGPUToCPU(Config<T>& config)
	{
		int size = static_cast<int>(config.updatedGrids.size());
		if (size > 0)
		{
			for (int s = 0; s < size; s++)
			{
				Grid<T>* grid = config.updatedGrids.at(s);
				ExchangeHelperGPUToCPU2(grid);
			}

			for (int s = 0; s < size; s++)
			{
				Grid<T>* grid = config.updatedGrids.at(s);
				ExchangeHelperGPUToCPU3(grid);
			}
		}
	}

	template<typename T>
	void ExchangeHelperGPUToCPU2(Grid<T>* grid)
	{
		int nDim = GetDimCount();
		bool periodic = IsPeriodic();
		int* blocks = GetBlocks();

		for (int i = 0; i < nDim; i++)
		{
			if (blocks[i] > 1 || periodic)
			{
				int face1 = LEFT;
				int face2 = RIGHT;

				if (i == 1)
				{
					if (nDim == 2)
					{
						face1 = UP;
						face2 = DOWN;
					}
					else
					{
						face1 = FRONT;
						face2 = BACK;
					}
				}
				else if (i == 2)
				{
					face1 = DOWN;
					face2 = UP;
				}

				int neighbour1 = GetNeighboursRank(face1);
				int neighbour2 = GetNeighboursRank(face2);

				if (neighbour1 != MPI_PROC_NULL)
				{
					CopyArrayToBufferGPUHelper(grid, face1);
				}

				if (neighbour2 != MPI_PROC_NULL)
				{
					CopyArrayToBufferGPUHelper(grid, face2);
				}
			}
		}
	}

	template<typename T>
	void ExchangeHelperGPUToCPU3(Grid<T>* grid)
	{
		int nDim = GetDimCount();
		bool periodic = IsPeriodic();
		int* blocks = GetBlocks();

		for (int i = 0; i < nDim; i++)
		{
			if (blocks[i] > 1 || periodic)
			{
				int face1 = LEFT;
				int face2 = RIGHT;

				if (i == 1)
				{
					if (nDim == 2)
					{
						face1 = UP;
						face2 = DOWN;
					}
					else
					{
						face1 = FRONT;
						face2 = BACK;
					}
				}
				else if (i == 2)
				{
					face1 = DOWN;
					face2 = UP;
				}

				int neighbour1 = GetNeighboursRank(face1);
				int neighbour2 = GetNeighboursRank(face2);

				if (neighbour1 != MPI_PROC_NULL)
				{
					grid->CopyHaloBufferGPUToCPU(face1);
				}

				if (neighbour2 != MPI_PROC_NULL)
				{
					grid->CopyHaloBufferGPUToCPU(face2);
				}
			}
		}
	}


	template<typename T>
	void ExchangeHelperGPU(Config<T>& config)
	{
		int size = static_cast<int>(config.updatedGrids.size());
		if (size > 0)
		{
			for (int s = 0; s < size; s++)
			{
				Grid<T>* grid = config.updatedGrids.at(s);
				Array<T>* arr = grid->GetSubGridCPU();
				HaloBuffer<T>* buffer = grid->GetBuffer();

				ExchangeGPU(arr, buffer);
			}
		}
	}


	template<typename T>
	void ExchangeHelperCPUToGPU(Config<T>& config)
	{
		int size = static_cast<int>(config.updatedGrids.size());
		if (size > 0)
		{
			for (int s = 0; s < size; s++)
			{
				Grid<T>* grid = config.updatedGrids.at(s);
				ExchangeHelperCPUToGPU2(grid);
			}

			for (int s = 0; s < size; s++)
			{
				Grid<T>* grid = config.updatedGrids.at(s);
				ExchangeHelperCPUToGPU3(grid);
			}
		}
	}


	template<typename T>
	void ExchangeHelperCPUToGPU2(Grid<T>* grid)
	{
		int nDim = GetDimCount();
		bool periodic = IsPeriodic();
		int* blocks = GetBlocks();

		for (int i = 0; i < nDim; i++)
		{
			if (blocks[i] > 1 || periodic)
			{
				int face1 = LEFT;
				int face2 = RIGHT;

				if (i == 1)
				{
					if (nDim == 2)
					{
						face1 = UP;
						face2 = DOWN;
					}
					else
					{
						face1 = FRONT;
						face2 = BACK;
					}
				}
				else if (i == 2)
				{
					face1 = DOWN;
					face2 = UP;
				}

				int neighbour1 = GetNeighboursRank(face1);
				int neighbour2 = GetNeighboursRank(face2);

				if (neighbour1 != MPI_PROC_NULL)
				{
					grid->CopyHaloBufferCPUToGPU(face1);
				}

				if (neighbour2 != MPI_PROC_NULL)
				{
					grid->CopyHaloBufferCPUToGPU(face2);
				}
			}
		}
	}


	template<typename T>
	void ExchangeHelperCPUToGPU3(Grid<T>* grid)
	{
		int nDim = GetDimCount();
		bool periodic = IsPeriodic();
		int* blocks = GetBlocks();

		for (int i = 0; i < nDim; i++)
		{
			if (blocks[i] > 1 || periodic)
			{
				int face1 = LEFT;
				int face2 = RIGHT;

				if (i == 1)
				{
					if (nDim == 2)
					{
						face1 = UP;
						face2 = DOWN;
					}
					else
					{
						face1 = FRONT;
						face2 = BACK;
					}
				}
				else if (i == 2)
				{
					face1 = DOWN;
					face2 = UP;
				}

				int neighbour1 = GetNeighboursRank(face1);
				int neighbour2 = GetNeighboursRank(face2);

				if (neighbour1 != MPI_PROC_NULL)
				{
					CopyBufferToArrayGPUHelper(grid, face1);
				}

				if (neighbour2 != MPI_PROC_NULL)
				{
					CopyBufferToArrayGPUHelper(grid, face2);
				}
			}
		}
	}


	template<typename... TArgs>
	__global__ void Kernel1D(int start, int end, TArgs... args)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		i = i + start;

		if (i < end)
		{
			STENCIL(i, args...);
		}
	}


	template<typename... TArgs>
	__global__ void Kernel2D(int start1, int end1, int start2, int end2, TArgs... args)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		i = i + start1;
		j = j + start2;

		if (i < end1 && j < end2)
		{
			STENCIL(i, j, args...);
		}
	}


	template<typename... TArgs>
	__global__ void Kernel3D(int start1, int end1, int start2, int end2, int start3, int end3, TArgs... args)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		i = i + start1;
		j = j + start2;
		k = k + start3;

		if (i < end1 && j < end2 && k < end3)
		{
			STENCIL(i, j, k, args...);
		}
	}


	template<typename T>
	__global__ void Kernel1DReduceSum(int start, int end, T* arr_i, T* arr_o)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		if (i >= start && i < end)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = 0;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				sdata[tx] += sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel2DReduceSum(int start1, int end1, int start2, int end2, int dimX, T* arr_i, T* arr_o)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		int ix = (i % dimX);
		int iy = (i / dimX);

		if (ix >= start1 && ix < end1 && iy >= start2 && iy < end2)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = 0;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				sdata[tx] += sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel3DReduceSum(int start1, int end1, int start2, int end2, int start3, int end3, int dimX, int dimY, T* arr_i, T* arr_o)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		int temp = (i % (dimX * dimY));
		int ix = (temp % dimX);
		int iy = (temp / dimX);
		int iz = (i / (dimX * dimY));

		if (ix >= start1 && ix < end1 && iy >= start2 && iy < end2 && iz >= start3 && iz < end3)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = 0;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				sdata[tx] += sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel1DReduceMax(int start, int end, T* arr_i, T* arr_o, T min)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		if (i >= start && i < end)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = min;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] > sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel2DReduceMax(int start1, int end1, int start2, int end2, int dimX, T* arr_i, T* arr_o, T min)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		int ix = (i % dimX);
		int iy = (i / dimX);

		if (ix >= start1 && ix < end1 && iy >= start2 && iy < end2)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = min;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] > sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel3DReduceMax(int start1, int end1, int start2, int end2, int start3, int end3, int dimX, int dimY, T* arr_i, T* arr_o, T min)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		int temp = (i % (dimX * dimY));
		int ix = (temp % dimX);
		int iy = (temp / dimX);
		int iz = (i / (dimX * dimY));

		if (ix >= start1 && ix < end1 && iy >= start2 && iy < end2 && iz >= start3 && iz < end3)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = min;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] > sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel1DReduceMin(int start, int end, T* arr_i, T* arr_o, T max)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		if (i >= start && i < end)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = max;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] < sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel2DReduceMin(int start1, int end1, int start2, int end2, int dimX, T* arr_i, T* arr_o, T max)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		int ix = (i % dimX);
		int iy = (i / dimX);

		if (ix >= start1 && ix < end1 && iy >= start2 && iy < end2)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = max;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] < sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void Kernel3DReduceMin(int start1, int end1, int start2, int end2, int start3, int end3, int dimX, int dimY, T* arr_i, T* arr_o, T max)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		int temp = (i % (dimX * dimY));
		int ix = (temp % dimX);
		int iy = (temp / dimX);
		int iz = (i / (dimX * dimY));

		if (ix >= start1 && ix < end1 && iy >= start2 && iy < end2 && iz >= start3 && iz < end3)
		{
			sdata[tx] = arr_i[i];
		}
		else
		{
			sdata[tx] = max;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] < sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr_o[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void KernelReduceSumLoop(int size, T* arr)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		if (i < size)
		{
			sdata[tx] = arr[i];
		}
		else
		{
			sdata[tx] = 0;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				sdata[tx] += sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void KernelReduceMaxLoop(int size, T* arr, T min)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		if (i < size)
		{
			sdata[tx] = arr[i];
		}
		else
		{
			sdata[tx] = min;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] > sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void KernelReduceMinLoop(int size, T* arr, T max)
	{
		__shared__ T sdata[64];
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int tx = threadIdx.x;

		if (i < size)
		{
			sdata[tx] = arr[i];
		}
		else
		{
			sdata[tx] = max;
		}

		__syncthreads();
		for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
		{
			if (tx < offset)
			{
				if (sdata[tx + offset] < sdata[tx]) sdata[tx] = sdata[tx + offset];
			}
			__syncthreads();
		}

		if (tx == 0)
		{
			arr[blockIdx.x] = sdata[0];
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer1D_LEFT(Array<T>* arr, T* buffer, int dimX, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;

		if (g < ghostLayer)
		{
			buffer[g] = arr->Get(ghostLayer + g);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer1D_RIGHT(Array<T>* arr, T* buffer, int dimX, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;

		if (g < ghostLayer)
		{
			buffer[g] = arr->Get(dimX - 2 * ghostLayer + g);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer2D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && j < dimY)
		{
			buffer[g * dimY + j] = arr->Get(ghostLayer + g, j);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer2D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && j < dimY)
		{
			buffer[g * dimY + j] = arr->Get(dimX - 2 * ghostLayer + g, j);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer2D_Y_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && i < dimX)
		{
			buffer[g * dimX + i] = arr->Get(i, ghostLayer + g);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer2D_Y_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && i < dimX)
		{
			buffer[g * dimX + i] = arr->Get(i, dimY - 2 * ghostLayer + g);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer3D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && j < dimY && k < dimZ)
		{
			buffer[g * dimY * dimZ + j + k * dimY] = arr->Get(ghostLayer + g, j, k);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer3D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && j < dimY && k < dimZ)
		{
			buffer[g * dimY * dimZ + j + k * dimY] = arr->Get(dimX - 2 * ghostLayer + g, j, k);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer3D_Y_FRONT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && k < dimZ)
		{
			buffer[g * dimX * dimZ + i + k * dimX] = arr->Get(i, ghostLayer + g, k);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer3D_Y_BACK(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && k < dimZ)
		{
			buffer[g * dimX * dimZ + i + k * dimX] = arr->Get(i, dimY - 2 * ghostLayer + g, k);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer3D_Z_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && j < dimY)
		{
			buffer[g * dimX * dimY + i + j * dimX] = arr->Get(i, j, ghostLayer + g);
		}
	}


	template<typename T>
	__global__ void ArrayToBuffer3D_Z_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && j < dimY)
		{
			buffer[g * dimX * dimY + i + j * dimX] = arr->Get(i, j, dimZ - 2 * ghostLayer + g);
		}
	}


	template<typename T>
	__global__ void BufferToArray1D_LEFT(Array<T>* arr, T* buffer, int dimX, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;

		if (g < ghostLayer)
		{
			arr->Set(g, buffer[g]);
		}
	}


	template<typename T>
	__global__ void BufferToArray1D_RIGHT(Array<T>* arr, T* buffer, int dimX, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;

		if (g < ghostLayer)
		{
			arr->Set(dimX - ghostLayer + g, buffer[g]);
		}
	}


	template<typename T>
	__global__ void BufferToArray2D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && j < dimY)
		{
			arr->Set(g, j, buffer[g * dimY + j]);
		}
	}


	template<typename T>
	__global__ void BufferToArray2D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && j < dimY)
		{
			arr->Set(dimX - ghostLayer + g, j, buffer[g * dimY + j]);
		}
	}


	template<typename T>
	__global__ void BufferToArray2D_Y_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && i < dimX)
		{
			arr->Set(i, g, buffer[g * dimX + i]);
		}
	}


	template<typename T>
	__global__ void BufferToArray2D_Y_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;

		if (g < ghostLayer && i < dimX)
		{
			arr->Set(i, dimY - ghostLayer + g, buffer[g * dimX + i]);
		}
	}


	template<typename T>
	__global__ void BufferToArray3D_X_LEFT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && j < dimY && k < dimZ)
		{
			arr->Set(g, j, k, buffer[g * dimY * dimZ + j + k * dimY]);
		}
	}


	template<typename T>
	__global__ void BufferToArray3D_X_RIGHT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && j < dimY && k < dimZ)
		{
			arr->Set(dimX - ghostLayer + g, j, k, buffer[g * dimY * dimZ + j + k * dimY]);
		}
	}


	template<typename T>
	__global__ void BufferToArray3D_Y_FRONT(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && k < dimZ)
		{
			arr->Set(i, g, k, buffer[g * dimX * dimZ + i + k * dimX]);
		}
	}


	template<typename T>
	__global__ void BufferToArray3D_Y_BACK(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && k < dimZ)
		{
			arr->Set(i, dimY - ghostLayer + g, k, buffer[g * dimX * dimZ + i + k * dimX]);
		}
	}


	template<typename T>
	__global__ void BufferToArray3D_Z_DOWN(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && j < dimY)
		{
			arr->Set(i, j, g, buffer[g * dimX * dimY + i + j * dimX]);
		}
	}


	template<typename T>
	__global__ void BufferToArray3D_Z_UP(Array<T>* arr, T* buffer, int dimX, int dimY, int dimZ, int ghostLayer)
	{
		int g = blockIdx.x * blockDim.x + threadIdx.x;
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.z * blockDim.z + threadIdx.z;

		if (g < ghostLayer && i < dimX && j < dimY)
		{
			arr->Set(i, j, dimZ - ghostLayer + g, buffer[g * dimX * dimY + i + j * dimX]);
		}
	}


}


#endif

#endif