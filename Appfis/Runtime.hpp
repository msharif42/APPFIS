/**
 *
 * This is the file containing methods that are useful in runtime
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef RUNTIME_HPP
#define RUNTIME_HPP


#include "Common.hpp"
#include "Array.hpp"
#include "HaloBuffer.hpp"
#include "Grid.hpp"
#include "Communication.hpp"
#include "Config.hpp"


namespace APPFIS
{
	template<typename T>
	void InitializeExecution(Grid<T>& grid);

	template<typename T, typename... TArgs>
	void InitializeExecution(Grid<T>& grid, TArgs&&... args);

	template<typename T>
	void FinalizeExecution(Grid<T>& grid);

	template<typename T, typename... TArgs>
	void FinalizeExecution(Grid<T>& grid, TArgs&&... args);

	template<typename T>
	void HaloExchange(Grid<T>* grid);

	template<typename T>
	void HaloExchangeCPU(Grid<T>* grid);

	template<typename T, typename... TArgs>
	void Execute1D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void Execute2D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void Execute3D(Config<T>& config, TArgs... args);

	template<typename T>
	void Execute1DReduce(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void Execute2DReduce(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void Execute3DReduce(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T, typename... TArgs>
	void Execute1DCPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void Execute2DCPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void Execute3DCPU(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteSerial1D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteSerial2D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteSerial3D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteOverlap1D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteOverlap2D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteOverlap3D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteBoundary1D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteBoundary2D(Config<T>& config, TArgs... args);

	template<typename T, typename... TArgs>
	void ExecuteBoundary3D(Config<T>& config, TArgs... args);

	template<typename T>
	void Execute1DReduceCPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void Execute2DReduceCPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void Execute3DReduceCPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid);

	template<typename T>
	void CopyFromArrayToBuffer(Array<T>* arr, HaloBuffer<T>* buffer, int face);

	template<typename T>
	void CopyFromBufferToArray(Array<T>* arr, HaloBuffer<T>* buffer, int face);

	template<typename T>
	void CopyFromArrayToBufferBorder(Array<T>* arr, HaloBuffer<T>* buffer, int face);

	template<typename T>
	void CopyFromBufferToArrayBorder(Array<T>* arr, HaloBuffer<T>* buffer, int face);

	template<typename T>
	void ExchangeHelper(Config<T>& config);

	template<typename T>
	void SwapHelper(Config<T>& config);


	template<typename T>
	void InitializeExecution(Grid<T>& grid)
	{
		EXECUTION_FLAG = true;

		if (GetGhostLayer() > 0 && grid.GetGridType() != FLAT)
		{
			ExchangeInitial(grid.GetSubGridCPU(), grid.GetBuffer());

			if (grid.GetGridType() == DUAL)
			{
				ExchangeInitial(grid.GetSubGridNextCPU(), grid.GetBuffer());
			}
		}

#ifdef APPFIS_CUDA
		grid.CreateSubGridsGPU();
#endif

	}


	template<typename T, typename... TArgs>
	void InitializeExecution(Grid<T>& grid, TArgs&&... args)
	{
		InitializeExecution(grid);
		InitializeExecution(args...);
	}


	template<typename T>
	void FinalizeExecution(Grid<T>& grid)
	{
		EXECUTION_FLAG = false;

#ifdef APPFIS_CUDA
		cudaStreamSynchronize(GetCudaStream());
		grid.CopySubGridsGPUToCPU();
#endif
	}


	template<typename T, typename... TArgs>
	void FinalizeExecution(Grid<T>& grid, TArgs&&... args)
	{
		FinalizeExecution(grid);
		FinalizeExecution(args...);
	}


	template<typename T>
	void HaloExchange(Grid<T>* grid)
	{
#ifdef APPFIS_CUDA
		HaloExchangeGPU(grid);
#else
		HaloExchangeCPU(grid);
#endif
	}


	template<typename T>
	void HaloExchangeCPU(Grid<T>* grid)
	{
		Exchange(grid->GetSubGridNextCPU(), grid->GetBuffer());
		grid->Swap();
	}


	template<typename T, typename... TArgs>
	void Execute1D(Config<T>& config, TArgs... args)
	{
#ifdef APPFIS_CUDA
		Execute1DGPU(config, args...);
#else
		Execute1DCPU(config, args...);
#endif
	}


	template<typename T, typename... TArgs>
	void Execute2D(Config<T>& config, TArgs... args)
	{
#ifdef APPFIS_CUDA
		Execute2DGPU(config, args...);
#else
		Execute2DCPU(config, args...);
#endif
	}


	template<typename T, typename... TArgs>
	void Execute3D(Config<T>& config, TArgs... args)
	{
#ifdef APPFIS_CUDA
		Execute3DGPU(config, args...);
#else
		Execute3DCPU(config, args...);
#endif
	}


	template<typename T>
	void Execute1DReduce(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
#ifdef APPFIS_CUDA
		Execute1DReduceGPU(config, var, rType, grid);
#else
		Execute1DReduceCPU(config, var, rType, grid);
#endif
	}


	template<typename T>
	void Execute2DReduce(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
#ifdef APPFIS_CUDA
		Execute2DReduceGPU(config, var, rType, grid);
#else
		Execute2DReduceCPU(config, var, rType, grid);
#endif
	}


	template<typename T>
	void Execute3DReduce(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
#ifdef APPFIS_CUDA
		Execute3DReduceGPU(config, var, rType, grid);
#else
		Execute3DReduceCPU(config, var, rType, grid);
#endif
	}


	template<typename T, typename... TArgs>
	void Execute1DCPU(Config<T>& config, TArgs... args)
	{
		bool overlap = IsOverlap();

		if (overlap && config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExecuteOverlap1D(config, args...);
		}
		else
		{
			ExecuteSerial1D(config, args...);
		}
	}


	template<typename T, typename... TArgs>
	void Execute2DCPU(Config<T>& config, TArgs... args)
	{
		bool overlap = IsOverlap();

		if (overlap && config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExecuteOverlap2D(config, args...);
		}
		else
		{
			ExecuteSerial2D(config, args...);
		}
	}


	template<typename T, typename... TArgs>
	void Execute3DCPU(Config<T>& config, TArgs... args)
	{
		bool overlap = IsOverlap();

		if (overlap && config.IsUpdated() && GetGhostLayer() > 0)
		{
			ExecuteOverlap3D(config, args...);
		}
		else
		{
			ExecuteSerial3D(config, args...);
		}
	}


	template<typename T, typename... TArgs>
	void ExecuteSerial1D(Config<T>& config, TArgs... args)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();

#pragma omp parallel for schedule(dynamic, 1)
		for (int i = startX; i < endX; i++)
		{
			STENCIL(i, args...);
		}

		ExchangeHelper(config);
		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteSerial2D(Config<T>& config, TArgs... args)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();

#pragma omp parallel for schedule(dynamic, 1)
		for (int j = startY; j < endY; j++)
		{
			for (int i = startX; i < endX; i++)
			{
				STENCIL(i, j, args...);
			}
		}

		ExchangeHelper(config);
		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteSerial3D(Config<T>& config, TArgs... args)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();
		int startZ = config.GetStartZ();
		int endZ = config.GetEndZ();

#pragma omp parallel for schedule(dynamic, 1)
		for (int k = startZ; k < endZ; k++)
		{
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
					STENCIL(i, j, k, args...);
				}
			}
		}

		ExchangeHelper(config);
		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteOverlap1D(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			if (tid == 0)
			{
				ExecuteBoundary1D(config, args...);
				ExchangeHelper(config);
			}

			int dimX = config.GetDimX();
			int startX = 2 * ghostLayer;
			int endX = dimX - 2 * ghostLayer;

#pragma omp for schedule(dynamic, 1)
			for (int i = startX; i < endX; i++)
			{
				STENCIL(i, args...);
			}
		}

		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteOverlap2D(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			if (tid == 0)
			{
				ExecuteBoundary2D(config, args...);
				ExchangeHelper(config);
			}

			int dimX = config.GetDimX();
			int dimY = config.GetDimY();
			int startX = 2 * ghostLayer;
			int endX = dimX - 2 * ghostLayer;
			int startY = 2 * ghostLayer;
			int endY = dimY - 2 * ghostLayer;

#pragma omp for schedule(dynamic, 1)
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
					STENCIL(i, j, args...);
				}
			}
		}

		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteOverlap3D(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			if (tid == 0)
			{
				ExecuteBoundary3D(config, args...);
				ExchangeHelper(config);
			}

			int dimX = config.GetDimX();
			int dimY = config.GetDimY();
			int dimZ = config.GetDimZ();
			int startX = 2 * ghostLayer;
			int endX = dimX - 2 * ghostLayer;
			int startY = 2 * ghostLayer;
			int endY = dimY - 2 * ghostLayer;
			int startZ = 2 * ghostLayer;
			int endZ = dimZ - 2 * ghostLayer;

#pragma omp for schedule(dynamic, 1)
			for (int k = startZ; k < endZ; k++)
			{
				for (int j = startY; j < endY; j++)
				{
					for (int i = startX; i < endX; i++)
					{
						STENCIL(i, j, k, args...);
					}
				}
			}
		}

		SwapHelper(config);
	}


	template<typename T, typename... TArgs>
	void ExecuteBoundary1D(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();
		int dimX = config.GetDimX();
		int startX = config.GetStartX();
		int endX = config.GetEndX();

		for (int i = startX; i < 2 * ghostLayer; i++)
		{
			STENCIL(i, args...);
		}

		for (int i = dimX - 2 * ghostLayer; i < endX; i++)
		{
			STENCIL(i, args...);
		}
	}


	template<typename T, typename... TArgs>
	void ExecuteBoundary2D(Config<T>& config, TArgs... args)
	{
		int ghostLayer = GetGhostLayer();
		int dimX = config.GetDimX();
		int dimY = config.GetDimY();
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();

		for (int j = startY; j < endY; j++)
		{
			for (int i = startX; i < 2 * ghostLayer; i++)
			{
				STENCIL(i, j, args...);
			}

			for (int i = dimX - 2 * ghostLayer; i < endX; i++)
			{
				STENCIL(i, j, args...);
			}
		}

		for (int i = 2 * ghostLayer; i < dimX - 2 * ghostLayer; i++)
		{
			for (int j = startY; j < 2 * ghostLayer; j++)
			{
				STENCIL(i, j, args...);
			}

			for (int j = dimY - 2 * ghostLayer; j < endY; j++)
			{
				STENCIL(i, j, args...);
			}
		}
	}


	template<typename T, typename... TArgs>
	void ExecuteBoundary3D(Config<T>& config, TArgs... args)
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

		for (int k = startZ; k < endZ; k++)
		{
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < 2 * ghostLayer; i++)
				{
					STENCIL(i, j, k, args...);
				}

				for (int i = dimX - 2 * ghostLayer; i < endX; i++)
				{
					STENCIL(i, j, k, args...);
				}
			}

			for (int i = 2 * ghostLayer; i < dimX - 2 * ghostLayer; i++)
			{
				for (int j = startY; j < 2 * ghostLayer; j++)
				{
					STENCIL(i, j, k, args...);
				}

				for (int j = dimY - 2 * ghostLayer; j < endY; j++)
				{
					STENCIL(i, j, k, args...);
				}
			}
		}

		for (int j = 2 * ghostLayer; j < dimY - 2 * ghostLayer; j++)
		{
			for (int i = 2 * ghostLayer; i < dimX - 2 * ghostLayer; i++)
			{
				for (int k = startZ; k < 2 * ghostLayer; k++)
				{
					STENCIL(i, j, k, args...);
				}

				for (int k = dimZ - 2 * ghostLayer; k < endZ; k++)
				{
					STENCIL(i, j, k, args...);
				}
			}
		}
	}


	template<typename T>
	void Execute1DReduceCPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();

		Array<T>* arr = grid.GetSubGridCPU();

		T local = 0;
		if (rType == MAX)
		{
			T max_val = (std::numeric_limits<T>::min)();
#ifdef _WIN32
#pragma omp parallel for
			for (int i = startX; i < endX; i++)
			{
#pragma omp critical
				if (arr->Get(i) > max_val)
				{
					max_val = arr->Get(i);
				}
			}
#else
#pragma omp parallel for reduction(max : max_val)
			for (int i = startX; i < endX; i++)
			{
				if (arr->Get(i) > max_val)
				{
					max_val = arr->Get(i);
				}
			}
#endif

			local = max_val;
		}
		else if (rType == MIN)
		{
			T min_val = (std::numeric_limits<T>::max)();
#ifdef _WIN32
#pragma omp parallel for
			for (int i = startX; i < endX; i++)
			{
#pragma omp critical
				if (arr->Get(i) < min_val)
				{
					min_val = arr > Get(i);
				}
			}
#else
#pragma omp parallel for reduction(min : min_val)
			for (int i = startX; i < endX; i++)
			{
				if (arr->Get(i) < min_val)
				{
					min_val = arr->Get(i);
				}
			}
#endif

			local = min_val;
		}
		else if (rType == SUM)
		{
			T sum_val = 0;
#ifdef _WIN32
#pragma omp parallel for
			for (int i = startX; i < endX; i++)
			{
#pragma omp critical
				sum_val += arr->Get(i);
			}
#else
#pragma omp parallel for reduction(+:sum_val)
			for (int i = startX; i < endX; i++)
			{
				sum_val += arr->Get(i);
			}
#endif

			local = sum_val;
		}

		var = AllReduce(local, rType);
	}


	template<typename T>
	void Execute2DReduceCPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();

		Array<T>* arr = grid.GetSubGridCPU();

		T local = 0;
		if (rType == MAX)
		{
			T max_val = (std::numeric_limits<T>::min)();
#ifdef _WIN32
#pragma omp parallel for
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
#pragma omp critical
					if (arr->Get(i, j) > max_val)
					{
						max_val = arr->Get(i, j);
					}
				}
			}
#else
#pragma omp parallel for reduction(max : max_val)
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
					if (arr->Get(i, j) > max_val)
					{
						max_val = arr->Get(i, j);
					}
				}
			}
#endif

			local = max_val;
		}
		else if (rType == MIN)
		{
			T min_val = (std::numeric_limits<T>::max)();

#ifdef _WIN32
#pragma omp parallel for
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
#pragma omp critical
					if (arr->Get(i, j) < min_val)
					{
						min_val = arr->Get(i, j);
					}
				}
			}
#else
#pragma omp parallel for reduction(min : min_val)
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
					if (arr->Get(i, j) < min_val)
					{
						min_val = arr->Get(i, j);
					}
				}
			}
#endif

			local = min_val;
		}
		else if (rType == SUM)
		{
			T sum_val = 0;
#ifdef _WIN32
#pragma omp parallel for
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
#pragma omp critical
					sum_val += arr->Get(i, j);
				}
			}
#else
#pragma omp parallel for reduction(+:sum_val)
			for (int j = startY; j < endY; j++)
			{
				for (int i = startX; i < endX; i++)
				{
					sum_val += arr->Get(i, j);
				}
			}
#endif

			local = sum_val;
		}

		var = AllReduce(local, rType);
	}


	template<typename T>
	void Execute3DReduceCPU(Config<T>& config, T& var, REDUCTION rType, Grid<T>& grid)
	{
		int startX = config.GetStartX();
		int endX = config.GetEndX();
		int startY = config.GetStartY();
		int endY = config.GetEndY();
		int startZ = config.GetStartZ();
		int endZ = config.GetEndZ();

		Array<T>* arr = grid.GetSubGridCPU();

		T local = 0;
		if (rType == MAX)
		{
			T max_val = (std::numeric_limits<T>::min)();
#ifdef _WIN32
#pragma omp parallel for
			for (int k = startZ; k < endZ; k++)
			{
				for (int j = startY; j < endY; j++)
				{
					for (int i = startX; i < endX; i++)
					{
#pragma omp critical
						if (arr->Get(i, j, k) > max_val)
						{
							max_val = arr->Get(i, j, k);
						}
					}
				}
			}
#else
#pragma omp parallel for reduction(max : max_val)
			for (int k = startZ; k < endZ; k++)
			{
				for (int j = startY; j < endY; j++)
				{
					for (int i = startX; i < endX; i++)
					{
						if (arr->Get(i, j, k) > max_val)
						{
							max_val = arr->Get(i, j, k);
						}
					}
				}
			}
#endif

			local = max_val;
		}
		else if (rType == MIN)
		{
			T min_val = (std::numeric_limits<T>::max)();
#ifdef _WIN32
#pragma omp parallel for
			for (int k = startZ; k < endZ; k++)
			{
				for (int j = startY; j < endY; j++)
				{
					for (int i = startX; i < endX; i++)
					{
#pragma omp critical
						if (arr->Get(i, j, k) < min_val)
						{
							min_val = arr->Get(i, j, k);
						}
					}
				}
			}
#else
#pragma omp parallel for reduction(min : min_val)
			for (int k = startZ; k < endZ; k++)
			{
				for (int j = startY; j < endY; j++)
				{
					for (int i = startX; i < endX; i++)
					{
						if (arr->Get(i, j, k) < min_val)
						{
							min_val = arr->Get(i, j, k);
						}
					}
				}
			}
#endif

			local = min_val;
		}
		else if (rType == SUM)
		{
			T sum_val = 0;
#ifdef _WIN32
#pragma omp parallel for
			for (int k = startZ; k < endZ; k++)
			{
				for (int j = startY; j < endY; j++)
				{
					for (int i = startX; i < endX; i++)
					{
#pragma omp critical
						sum_val += arr->Get(i, j, k);
					}
				}
			}
#else
#pragma omp parallel for reduction(+:sum_val)
			for (int k = startZ; k < endZ; k++)
			{
				for (int j = startY; j < endY; j++)
				{
					for (int i = startX; i < endX; i++)
					{
						sum_val += arr->Get(i, j, k);
					}
				}
			}
#endif

			local = sum_val;
		}

		var = AllReduce(local, rType);
	}


	template<typename T>
	void CopyFromArrayToBuffer(Array<T>* arr, HaloBuffer<T>* buffer, int face)
	{
		if (face >= buffer->GetFaceCount())
		{
			std::cerr << "Invalid face of the data" << std::endl;
			return;
		}

		int ghostLayer = GetGhostLayer();
		for (int g = 0; g < ghostLayer; g++)
		{
			if (arr->GetDimCount() == DIM_1D)
			{
				int dimX = arr->GetDim(X);

				if (face == LEFT)
				{
					buffer->Set(LEFT, g, arr->Get(ghostLayer + g));
				}
				else if (face == RIGHT)
				{
					buffer->Set(RIGHT, g, arr->Get(dimX - 2 * ghostLayer + g));
				}
			}
			else if (arr->GetDimCount() == DIM_2D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);

				if (face == LEFT)
				{
					for (int j = 0; j < dimY; j++)
					{
						buffer->Set(LEFT, g * dimY + j, arr->Get(ghostLayer + g, j));
					}
				}
				else if (face == RIGHT)
				{
					for (int j = 0; j < dimY; j++)
					{
						buffer->Set(RIGHT, g * dimY + j, arr->Get(dimX - 2 * ghostLayer + g, j));
					}
				}
				else if (face == UP)
				{
					for (int i = 0; i < dimX; i++)
					{
						buffer->Set(UP, g * dimX + i, arr->Get(i, ghostLayer + g));
					}
				}
				else if (face == DOWN)
				{
					for (int i = 0; i < dimX; i++)
					{
						buffer->Set(DOWN, g * dimX + i, arr->Get(i, dimY - 2 * ghostLayer + g));
					}
				}
			}
			else if (arr->GetDimCount() == DIM_3D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);
				int dimZ = arr->GetDim(Z);

				if (face == LEFT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int j = 0; j < dimY; j++)
						{
							buffer->Set(LEFT, g * dimY * dimZ + j + k * dimY, arr->Get(ghostLayer + g, j, k));
						}
					}
				}
				else if (face == RIGHT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int j = 0; j < dimY; j++)
						{
							buffer->Set(RIGHT, g * dimY * dimZ + j + k * dimY, arr->Get(dimX - 2 * ghostLayer + g, j, k));
						}
					}
				}
				else if (face == FRONT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							buffer->Set(FRONT, g * dimX * dimZ + i + k * dimX, arr->Get(i, ghostLayer + g, k));
						}
					}
				}
				else if (face == BACK)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							buffer->Set(BACK, g * dimX * dimZ + i + k * dimX, arr->Get(i, dimY - 2 * ghostLayer + g, k));
						}
					}
				}
				else if (face == DOWN)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							buffer->Set(DOWN, g * dimX * dimY + i + j * dimX, arr->Get(i, j, ghostLayer + g));
						}
					}
				}
				else if (face == UP)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							buffer->Set(UP, g * dimX * dimY + i + j * dimX, arr->Get(i, j, dimZ - 2 * ghostLayer + g));
						}
					}
				}
			}
		}
	}


	template<typename T>
	void CopyFromBufferToArray(Array<T>* arr, HaloBuffer<T>* buffer, int face)
	{
		if (face >= buffer->GetFaceCount())
		{
			std::cerr << "Invalid face of the data" << std::endl;
			return;
		}

		int ghostLayer = GetGhostLayer();
		for (int g = 0; g < ghostLayer; g++)
		{
			if (arr->GetDimCount() == DIM_1D)
			{
				int dimX = arr->GetDim(X);

				if (face == LEFT)
				{
					arr->Set(g, buffer->Get(LEFT, g));
				}
				else if (face == RIGHT)
				{
					arr->Set(dimX - ghostLayer + g, buffer->Get(RIGHT, g));
				}
			}
			else if (arr->GetDimCount() == DIM_2D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);

				if (face == LEFT)
				{
					for (int j = 0; j < dimY; j++)
					{
						arr->Set(g, j, buffer->Get(LEFT, g * dimY + j));
					}
				}
				else if (face == RIGHT)
				{
					for (int j = 0; j < dimY; j++)
					{
						arr->Set(dimX - ghostLayer + g, j, buffer->Get(RIGHT, g * dimY + j));
					}
				}
				else if (face == UP)
				{
					for (int i = 0; i < dimX; i++)
					{
						arr->Set(i, g, buffer->Get(UP, g * dimX + i));
					}
				}
				else if (face == DOWN)
				{
					for (int i = 0; i < dimX; i++)
					{
						arr->Set(i, dimY - ghostLayer + g, buffer->Get(DOWN, g * dimX + i));
					}
				}
			}
			else if (arr->GetDimCount() == DIM_3D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);
				int dimZ = arr->GetDim(Z);

				if (face == LEFT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int j = 0; j < dimY; j++)
						{
							arr->Set(g, j, k, buffer->Get(LEFT, g * dimY * dimZ + j + k * dimY));
						}
					}
				}
				else if (face == RIGHT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int j = 0; j < dimY; j++)
						{
							arr->Set(dimX - ghostLayer + g, j, k, buffer->Get(RIGHT, g * dimY * dimZ + j + k * dimY));
						}
					}
				}
				else if (face == FRONT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, g, k, buffer->Get(FRONT, g * dimX * dimZ + i + k * dimX));
						}
					}
				}
				else if (face == BACK)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, dimY - ghostLayer + g, k, buffer->Get(BACK, g * dimX * dimZ + i + k * dimX));
						}
					}
				}
				else if (face == DOWN)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, j, g, buffer->Get(DOWN, g * dimX * dimY + i + j * dimX));
						}
					}
				}
				else if (face == UP)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, j, dimZ - ghostLayer + g, buffer->Get(UP, g * dimX * dimY + i + j * dimX));
						}
					}
				}
			}
		}
	}


	template<typename T>
	void CopyFromArrayToBufferBorder(Array<T>* arr, HaloBuffer<T>* buffer, int face)
	{
		if (face >= buffer->GetFaceCount())
		{
			std::cerr << "Invalid face of the data" << std::endl;
			return;
		}

		int ghostLayer = GetGhostLayer();
		for (int g = 0; g < ghostLayer; g++)
		{
			if (arr->GetDimCount() == DIM_2D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);

				if (face == UP)
				{
					int i = 0;
					buffer->Set(UP, g * dimX + i, arr->Get(i, ghostLayer + g));

					i = dimX - 1;
					buffer->Set(UP, g * dimX + i, arr->Get(i, ghostLayer + g));
				}
				else if (face == DOWN)
				{
					int i = 0;
					buffer->Set(DOWN, g * dimX + i, arr->Get(i, dimY - 2 * ghostLayer + g));

					i = dimX - 1;
					buffer->Set(DOWN, g * dimX + i, arr->Get(i, dimY - 2 * ghostLayer + g));
				}
			}
			else if (arr->GetDimCount() == DIM_3D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);
				int dimZ = arr->GetDim(Z);

				if (face == FRONT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						int i = 0;
						buffer->Set(FRONT, g * dimX * dimZ + i + k * dimX, arr->Get(i, ghostLayer + g, k));

						i = dimX - 1;
						buffer->Set(FRONT, g * dimX * dimZ + i + k * dimX, arr->Get(i, ghostLayer + g, k));
					}
				}
				else if (face == BACK)
				{
					for (int k = 0; k < dimZ; k++)
					{
						int i = 0;
						buffer->Set(BACK, g * dimX * dimZ + i + k * dimX, arr->Get(i, dimY - 2 * ghostLayer + g, k));

						i = dimX - 1;
						buffer->Set(BACK, g * dimX * dimZ + i + k * dimX, arr->Get(i, dimY - 2 * ghostLayer + g, k));
					}
				}
				else if (face == DOWN)
				{
					for (int j = 0; j < dimY; j++)
					{
						int i = 0;
						buffer->Set(DOWN, g * dimX * dimY + i + j * dimX, arr->Get(i, j, ghostLayer + g));

						i = dimX - 1;
						buffer->Set(DOWN, g * dimX * dimY + i + j * dimX, arr->Get(i, j, ghostLayer + g));
					}
				}
				else if (face == UP)
				{
					for (int j = 0; j < dimY; j++)
					{
						int i = 0;
						buffer->Set(UP, g * dimX * dimY + i + j * dimX, arr->Get(i, j, dimZ - 2 * ghostLayer + g));

						i = dimX - 1;
						buffer->Set(UP, g * dimX * dimY + i + j * dimX, arr->Get(i, j, dimZ - 2 * ghostLayer + g));
					}
				}
			}
		}
	}


	template<typename T>
	void CopyFromBufferToArrayBorder(Array<T>* arr, HaloBuffer<T>* buffer, int face)
	{
		if (face >= buffer->GetFaceCount())
		{
			std::cerr << "Invalid face of the data" << std::endl;
			return;
		}

		int ghostLayer = GetGhostLayer();
		for (int g = 0; g < ghostLayer; g++)
		{
			if (arr->GetDimCount() == DIM_2D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);

				if (face == LEFT)
				{
					int j = ghostLayer;
					arr->Set(g, j, buffer->Get(LEFT, g * dimY + j));

					j = dimY - 1 - ghostLayer;
					arr->Set(g, j, buffer->Get(LEFT, g * dimY + j));
				}
				else if (face == RIGHT)
				{
					int j = ghostLayer;
					arr->Set(dimX - ghostLayer + g, j, buffer->Get(RIGHT, g * dimY + j));

					j = dimY - 1 - ghostLayer;
					arr->Set(dimX - ghostLayer + g, j, buffer->Get(RIGHT, g * dimY + j));
				}
				else if (face == UP)
				{
					int i = ghostLayer;
					arr->Set(i, g, buffer->Get(UP, g * dimX + i));

					i = dimX - 1 - ghostLayer;
					arr->Set(i, g, buffer->Get(UP, g * dimX + i));
				}
				else if (face == DOWN)
				{
					int i = ghostLayer;
					arr->Set(i, dimY - ghostLayer + g, buffer->Get(DOWN, g * dimX + i));

					i = dimX - 1 - ghostLayer;
					arr->Set(i, dimY - ghostLayer + g, buffer->Get(DOWN, g * dimX + i));
				}
			}
			else if (arr->GetDimCount() == DIM_3D)
			{
				int dimX = arr->GetDim(X);
				int dimY = arr->GetDim(Y);
				int dimZ = arr->GetDim(Z);

				if (face == LEFT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						int j = ghostLayer;
						arr->Set(g, j, k, buffer->Get(LEFT, g * dimY * dimZ + j + k * dimY));

						j = dimY - 1 - ghostLayer;
						arr->Set(g, j, k, buffer->Get(LEFT, g * dimY * dimZ + j + k * dimY));
					}
				}
				else if (face == RIGHT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						int j = ghostLayer;
						arr->Set(dimX - ghostLayer + g, j, k, buffer->Get(RIGHT, g * dimY * dimZ + j + k * dimY));

						j = dimY - 1 - ghostLayer;
						arr->Set(dimX - ghostLayer + g, j, k, buffer->Get(RIGHT, g * dimY * dimZ + j + k * dimY));
					}
				}
				else if (face == FRONT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						int i = ghostLayer;
						arr->Set(i, g, k, buffer->Get(FRONT, g * dimX * dimZ + i + k * dimX));

						i = dimX - 1 - ghostLayer;
						arr->Set(i, g, k, buffer->Get(FRONT, g * dimX * dimZ + i + k * dimX));
					}
				}
				else if (face == BACK)
				{
					for (int k = 0; k < dimZ; k++)
					{
						int i = ghostLayer;
						arr->Set(i, dimY - ghostLayer + g, k, buffer->Get(BACK, g * dimX * dimZ + i + k * dimX));

						i = dimX - 1 - ghostLayer;
						arr->Set(i, dimY - ghostLayer + g, k, buffer->Get(BACK, g * dimX * dimZ + i + k * dimX));
					}
				}
				else if (face == DOWN)
				{
					for (int j = 0; j < dimY; j++)
					{
						int i = ghostLayer;
						arr->Set(i, j, g, buffer->Get(DOWN, g * dimX * dimY + i + j * dimX));

						i = dimX - 1 - ghostLayer;
						arr->Set(i, j, g, buffer->Get(DOWN, g * dimX * dimY + i + j * dimX));
					}
				}
				else if (face == UP)
				{
					for (int j = 0; j < dimY; j++)
					{
						int i = ghostLayer;
						arr->Set(i, j, dimZ - ghostLayer + g, buffer->Get(UP, g * dimX * dimY + i + j * dimX));

						i = dimX - 1 - ghostLayer;
						arr->Set(i, j, dimZ - ghostLayer + g, buffer->Get(UP, g * dimX * dimY + i + j * dimX));
					}
				}
			}
		}
	}


	template<typename T>
	void ExchangeHelper(Config<T>& config)
	{
		int size = static_cast<int>(config.updatedGrids.size());
		if (size > 0)
		{
			for (int i = 0; i < size; i++)
			{
				Grid<T>* grid = config.updatedGrids.at(i);
				Exchange(grid->GetSubGridNextCPU(), grid->GetBuffer());
			}
		}
	}


	template<typename T>
	void SwapHelper(Config<T>& config)
	{
		int size = static_cast<int>(config.updatedGrids.size());
		if (size > 0)
		{
			for (int i = 0; i < size; i++)
			{
				Grid<T>* grid = config.updatedGrids.at(i);
				grid->Swap();
			}
		}
	}

}


#endif