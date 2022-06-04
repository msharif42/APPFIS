/**
 *
 * This is the main data structure class.
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef GRID_HPP
#define GRID_HPP


#include "Env.hpp"
#include "Common.hpp"
#include "Array.hpp"
#include "HaloBuffer.hpp"


namespace APPFIS
{

	template<class T>
	class Grid
	{

	private:
		int _nDim = 0;
		int _currIndex = 0;
		int _nextIndex = 0;

		Array<T> _full;
		std::vector<Array<T>> _holder;

		HaloBuffer<T> _haloBuffer;

#ifdef APPFIS_CUDA
		std::vector<Array<T>*> _holder_gpu;
		std::vector<T*> _holder_gpu_data;
		std::vector<int*> _holder_gpu_dims;

		std::vector<T*> _haloBuffer_gpu_send;
		std::vector<T*> _haloBuffer_gpu_recv;
#endif

		GRID_TYPE _gridType = SINGLE;


	public:
		Grid<T>();
		~Grid<T>();

		Grid<T>(int nDim, int dimX, GRID_TYPE gridType);
		Grid<T>(int nDim, int dimX, int dimY, GRID_TYPE gridType);
		Grid<T>(int nDim, int dimX, int dimY, int dimZ, GRID_TYPE gridType);
		Grid<T>(int nDim, int* dims, GRID_TYPE gridType);

		void Create(int nDim, int dimX, GRID_TYPE gridType);
		void Create(int nDim, int dimX, int dimY, GRID_TYPE gridType);
		void Create(int nDim, int dimX, int dimY, int dimZ, GRID_TYPE gridType);
		void Create(int nDim, int* dims, GRID_TYPE gridType);

		Grid<T>(const Grid<T>& grid);
		Grid& operator=(const Grid<T>& grid);

		int GetDimCount() const;

		Array<T>* GetFullGrid();
		Array<T>* GetSubGrid();
		Array<T>* GetSubGridNext();
		Array<T>* GetSubGridCPU();
		Array<T>* GetSubGridNextCPU();

		GRID_TYPE GetGridType() const;
		HaloBuffer<T>* GetBuffer();

		void AllocateFullGrid();
		void CreateSubGrids();
		void SetBoundary(T value);
		void SetBoundary(int face, T value);
		void SetBoundary(Array<T>* arr, int face, T value);
		void SetBoundaryCopy();
		void SetBoundaryCopy(int face);
		void SetBoundaryCopy(Array<T>* arr, int face);
		void Reset();
		void Swap();

#ifdef APPFIS_CUDA
		void CreateSubGridsGPU();
		void CopySubGridsGPUToCPU();
		void CopyHaloBufferGPUToCPU(int face);
		void CopyHaloBufferCPUToGPU(int face);
		T* GetSendBufferGPU(int face);
		T* GetRecvBufferGPU(int face);
		T* GetSubGridsGPUData();
		T* GetSubGridsNextGPUData();
#endif

	};


	template<class T>
	Grid<T>::Grid()
	{
	}


	template<class T>
	Grid<T>::~Grid()
	{

#ifdef APPFIS_CUDA
		while (!_holder_gpu_data.empty())
		{
			cudaFree(_holder_gpu_data.back());
			_holder_gpu_data.pop_back();
		}
		while (!_holder_gpu_dims.empty())
		{
			cudaFree(_holder_gpu_dims.back());
			_holder_gpu_dims.pop_back();
		}
		while (!_holder_gpu.empty())
		{
			cudaFree(_holder_gpu.back());
			_holder_gpu.pop_back();
		}
		while (!_haloBuffer_gpu_send.empty())
		{
			cudaFree(_haloBuffer_gpu_send.back());
			_haloBuffer_gpu_send.pop_back();
		}
		while (!_haloBuffer_gpu_recv.empty())
		{
			cudaFree(_haloBuffer_gpu_recv.back());
			_haloBuffer_gpu_recv.pop_back();
		}
#endif

	}


	template<class T>
	Grid<T>::Grid(int nDim, int dimX, GRID_TYPE gridType)
	{
		Create(nDim, dimX, gridType);
	}


	template<class T>
	Grid<T>::Grid(int nDim, int dimX, int dimY, GRID_TYPE gridType)
	{
		Create(nDim, dimX, dimY, gridType);
	}


	template<class T>
	Grid<T>::Grid(int nDim, int dimX, int dimY, int dimZ, GRID_TYPE gridType)
	{
		Create(nDim, dimX, dimY, dimZ, gridType);
	}


	template<class T>
	Grid<T>::Grid(int nDim, int* dims, GRID_TYPE gridType)
	{
		Create(nDim, dims, gridType);
	}


	template<class T>
	Grid<T>::Grid(const Grid<T>& grid)
	{
		std::cout << "Grid copy" << std::endl;

		this->Reset();

		if (grid._nDim == 0)
		{
			return;
		}

		this->_nDim = grid._nDim;
		this->_gridType = grid._gridType;

		this->_full = grid._full;

		if (this->_holder.size() == 0)
		{
			this->_holder = std::vector<Array<T>>(2);
		}

		this->_currIndex = grid._currIndex;
		this->_nextIndex = grid._nextIndex;
		this->_holder.at(this->_currIndex) = grid._holder.at(this->_currIndex);
		this->_holder.at(this->_nextIndex) = grid._holder.at(this->_nextIndex);

		this->_haloBuffer = grid._haloBuffer;
	}


	template<typename T>
	Grid<T>& Grid<T>::operator=(const Grid<T>& grid)
	{
		std::cout << "Grid assign" << std::endl;

		this->Reset();

		if (grid._nDim == 0)
		{
			return *this;
		}

		this->_nDim = grid._nDim;
		this->_gridType = grid._gridType;

		this->_full = grid._full;

		if (this->_holder.size() == 0)
		{
			this->_holder = std::vector<Array<T>>(2);
		}

		this->_currIndex = grid._currIndex;
		this->_nextIndex = grid._nextIndex;
		this->_holder.at(this->_currIndex) = grid._holder.at(this->_currIndex);
		this->_holder.at(this->_nextIndex) = grid._holder.at(this->_nextIndex);

		this->_haloBuffer = grid._haloBuffer;

		return *this;
	}


	template<typename T>
	void Grid<T>::Create(int nDim, int dimX, GRID_TYPE gridType)
	{
		this->_nDim = nDim;
		this->_gridType = gridType;

		int dims[1] = { dimX };
		this->_full.Resize(nDim, dims);
		this->CreateSubGrids();
	}


	template<typename T>
	void Grid<T>::Create(int nDim, int dimX, int dimY, GRID_TYPE gridType)
	{
		this->_nDim = nDim;
		this->_gridType = gridType;

		int dims[2] = { dimX, dimY };
		this->_full.Resize(nDim, dims);
		this->CreateSubGrids();
	}


	template<typename T>
	void Grid<T>::Create(int nDim, int dimX, int dimY, int dimZ, GRID_TYPE gridType)
	{
		this->_nDim = nDim;
		this->_gridType = gridType;

		int dims[3] = { dimX, dimY, dimZ };
		this->_full.Resize(nDim, dims);
		this->CreateSubGrids();
	}


	template<typename T>
	void Grid<T>::Create(int nDim, int* dims, GRID_TYPE gridType)
	{
		this->_nDim = nDim;
		this->_gridType = gridType;

		this->_full.Resize(nDim, dims);
		this->CreateSubGrids();
	}


	template<typename T>
	int Grid<T>::GetDimCount() const
	{
		return this->_nDim;
	}


	template<typename T>
	Array<T>* Grid<T>::GetFullGrid()
	{
		return &this->_full;
	}


	template<typename T>
	Array<T>* Grid<T>::GetSubGrid()
	{
		if (!EXECUTION_FLAG)
		{
			return &this->_holder.at(this->_currIndex);
		}
		else
		{
#ifdef APPFIS_CUDA
			return this->_holder_gpu.at(this->_currIndex);
#else
			return &this->_holder.at(this->_currIndex);
#endif
		}
	}


	template<typename T>
	Array<T>* Grid<T>::GetSubGridNext()
	{
		if (!EXECUTION_FLAG)
		{
			return &this->_holder.at(this->_nextIndex);
		}
		else
		{
#ifdef APPFIS_CUDA
			return this->_holder_gpu.at(this->_nextIndex);
#else
			return &this->_holder.at(this->_nextIndex);
#endif
		}
	}


	template<typename T>
	Array<T>* Grid<T>::GetSubGridCPU()
	{
		return &this->_holder.at(this->_currIndex);
	}


	template<typename T>
	Array<T>* Grid<T>::GetSubGridNextCPU()
	{
		return &this->_holder.at(this->_nextIndex);
	}


	template<typename T>
	GRID_TYPE Grid<T>::GetGridType() const
	{
		return this->_gridType;
	}


	template<typename T>
	HaloBuffer<T>* Grid<T>::GetBuffer()
	{
		return &this->_haloBuffer;
	}


	template<typename T>
	void Grid<T>::AllocateFullGrid()
	{
		this->_full.AllocateMemory();
	}


	template<typename T>
	void Grid<T>::CreateSubGrids()
	{
		this->_holder = std::vector<Array<T>>(2);

		int* subDims = new int[this->_nDim];
		if (this->_gridType == FLAT)
		{
			for (int i = 0; i < this->_nDim; i++)
			{
				subDims[i] = this->_full.GetDims()[i];
			}
		}
		else
		{
			GetLocalPartitionDims(this->_nDim, this->_full.GetDims(), subDims);
		}

		this->_holder.at(this->_currIndex).Resize(this->_nDim, subDims);
		this->_holder.at(this->_currIndex).AllocateMemory();

		if (this->_gridType == DUAL)
		{
			this->_nextIndex = 1;
			this->_holder.at(this->_nextIndex).Resize(this->_nDim, subDims);
			this->_holder.at(this->_nextIndex).AllocateMemory();
		}

		this->_haloBuffer.Create(this->_nDim, subDims);

		delete[] subDims;
	}


#ifdef APPFIS_CUDA

	template<typename T>
	void Grid<T>::CreateSubGridsGPU()
	{
		Array<T>* a = &this->_holder.at(this->_currIndex);
		Array<T>* d_a;

		cudaMalloc((void**)&d_a, sizeof(Array<T>));
		cudaMemcpy(d_a, a, sizeof(Array<T>), cudaMemcpyHostToDevice);

		T* hostdata;
		cudaMalloc((void**)&hostdata, a->GetSize() * sizeof(T));
		cudaMemcpy(hostdata, a->GetData(), a->GetSize() * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a->DataAddress(), &hostdata, sizeof(T*), cudaMemcpyHostToDevice);

		int* hostdims;
		cudaMalloc((void**)&hostdims, a->GetDimCount() * sizeof(int));
		cudaMemcpy(hostdims, a->GetDims(), a->GetDimCount() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_a->DimsAddress(), &hostdims, sizeof(int*), cudaMemcpyHostToDevice);

		this->_holder_gpu.push_back(d_a);
		this->_holder_gpu_data.push_back(hostdata);
		this->_holder_gpu_dims.push_back(hostdims);

		if (this->_gridType == DUAL)
		{
			Array<T>* b = &this->_holder.at(this->_nextIndex);
			Array<T>* d_b;

			cudaMalloc((void**)&d_b, sizeof(Array<T>));
			cudaMemcpy(d_b, b, sizeof(Array<T>), cudaMemcpyHostToDevice);

			T* hostdata2;
			cudaMalloc((void**)&hostdata2, b->GetSize() * sizeof(T));
			cudaMemcpy(hostdata2, b->GetData(), b->GetSize() * sizeof(T), cudaMemcpyHostToDevice);
			cudaMemcpy(d_b->DataAddress(), &hostdata2, sizeof(T*), cudaMemcpyHostToDevice);

			int* hostdims2;
			cudaMalloc((void**)&hostdims2, b->GetDimCount() * sizeof(int));
			cudaMemcpy(hostdims2, b->GetDims(), b->GetDimCount() * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_b->DimsAddress(), &hostdims2, sizeof(int*), cudaMemcpyHostToDevice);

			this->_holder_gpu.push_back(d_b);
			this->_holder_gpu_data.push_back(hostdata2);
			this->_holder_gpu_dims.push_back(hostdims2);
		}

		HaloBuffer<T>* hostBuffer = &this->_haloBuffer;

		int face = hostBuffer->GetFaceCount();
		for (int i = 0; i < face; i++)
		{
			T* _hostSendBuffer;
			cudaMalloc((void**)&_hostSendBuffer, hostBuffer->GetHaloSize(i) * sizeof(T));
			cudaMemcpy(_hostSendBuffer, hostBuffer->GetSendBuffer(i), hostBuffer->GetHaloSize(i) * sizeof(T), cudaMemcpyHostToDevice);

			this->_haloBuffer_gpu_send.push_back(_hostSendBuffer);

			T* _hostRecvBuffer;
			cudaMalloc((void**)&_hostRecvBuffer, hostBuffer->GetHaloSize(i) * sizeof(T));
			cudaMemcpy(_hostRecvBuffer, hostBuffer->GetRecvBuffer(i), hostBuffer->GetHaloSize(i) * sizeof(T), cudaMemcpyHostToDevice);

			this->_haloBuffer_gpu_recv.push_back(_hostRecvBuffer);
		}

		cudaDeviceSynchronize();
	}


	template<typename T>
	void Grid<T>::CopySubGridsGPUToCPU()
	{
		Array<T>* a = &this->_holder.at(this->_currIndex);
		T* hostdata = this->_holder_gpu_data.at(this->_currIndex);

		cudaMemcpy(a->GetData(), hostdata, a->GetSize() * sizeof(T), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
	}


	template<typename T>
	void Grid<T>::CopyHaloBufferGPUToCPU(int face)
	{
		HaloBuffer<T>* hostBuffer = &this->_haloBuffer;
		T* _hostSendBuffer = this->_haloBuffer_gpu_send.at(face);
		cudaMemcpyAsync(hostBuffer->GetSendBuffer(face), _hostSendBuffer, hostBuffer->GetHaloSize(face) * sizeof(T), cudaMemcpyDeviceToHost, GetCudaStream());
	}


	template<typename T>
	void Grid<T>::CopyHaloBufferCPUToGPU(int face)
	{
		HaloBuffer<T>* hostBuffer = &this->_haloBuffer;
		T* _hostRecvBuffer = this->_haloBuffer_gpu_recv.at(face);
		cudaMemcpyAsync(_hostRecvBuffer, hostBuffer->GetRecvBuffer(face), hostBuffer->GetHaloSize(face) * sizeof(T), cudaMemcpyHostToDevice, GetCudaStream());
	}


	template<typename T>
	T* Grid<T>::GetSendBufferGPU(int face)
	{
		return this->_haloBuffer_gpu_send.at(face);
	}


	template<typename T>
	T* Grid<T>::GetRecvBufferGPU(int face)
	{
		return this->_haloBuffer_gpu_recv.at(face);
	}


	template<typename T>
	T* Grid<T>::GetSubGridsGPUData()
	{
		return this->_holder_gpu_data.at(this->_currIndex);
	}

	template<typename T>
	T* Grid<T>::GetSubGridsNextGPUData()
	{
		return this->_holder_gpu_data.at(this->_nextIndex);
	}

#endif


	template<typename T>
	void Grid<T>::SetBoundary(T value)
	{
		int nFace = this->_haloBuffer.GetFaceCount();
		for (int i = 0; i < nFace; i++)
		{
			this->SetBoundary(&this->_holder.at(this->_currIndex), i, value);

			if (this->_gridType == DUAL)
			{
				this->SetBoundary(&this->_holder.at(this->_nextIndex), i, value);
			}
		}
	}


	template<typename T>
	void Grid<T>::SetBoundary(int face, T value)
	{
		this->SetBoundary(&this->_holder.at(this->_currIndex), face, value);

		if (this->_gridType == DUAL)
		{
			this->SetBoundary(&this->_holder.at(this->_nextIndex), face, value);
		}
	}


	template<typename T>
	void Grid<T>::SetBoundary(Array<T>* arr, int face, T value)
	{
		int ghostLayer = GetGhostLayer();
		for (int g = 0; g < ghostLayer; g++)
		{
			if (arr->GetDimCount() == DIM_1D)
			{
				int dimX = arr->GetDim(X);

				if (face == LEFT)
				{
					arr->Set(g, value);
				}
				else if (face == RIGHT)
				{
					arr->Set(dimX - ghostLayer + g, value);
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
						arr->Set(g, j, value);
					}
				}
				else if (face == RIGHT)
				{
					for (int j = 0; j < dimY; j++)
					{
						arr->Set(dimX - ghostLayer + g, j, value);
					}
				}
				else if (face == UP)
				{
					for (int i = 0; i < dimX; i++)
					{
						arr->Set(i, g, value);
					}
				}
				else if (face == DOWN)
				{
					for (int i = 0; i < dimX; i++)
					{
						arr->Set(i, dimY - ghostLayer + g, value);
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
							arr->Set(g, j, k, value);
						}
					}
				}
				else if (face == RIGHT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int j = 0; j < dimY; j++)
						{
							arr->Set(dimX - ghostLayer + g, j, k, value);
						}
					}
				}
				else if (face == FRONT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, g, k, value);
						}
					}
				}
				else if (face == BACK)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, dimY - ghostLayer + g, k, value);
						}
					}
				}
				else if (face == DOWN)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, j, g, value);
						}
					}
				}
				else if (face == UP)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, j, dimZ - ghostLayer + g, value);
						}
					}
				}
			}
		}
	}


	template<typename T>
	void Grid<T>::SetBoundaryCopy()
	{
		int nFace = this->_haloBuffer.GetFaceCount();
		for (int i = 0; i < nFace; i++)
		{
			this->SetBoundaryCopy(&this->_holder.at(this->_currIndex), i);

			if (this->_gridType == DUAL)
			{
				this->SetBoundaryCopy(&this->_holder.at(this->_nextIndex), i);
			}
		}
	}


	template<typename T>
	void Grid<T>::SetBoundaryCopy(int face)
	{
		this->SetBoundaryCopy(&this->_holder.at(this->_currIndex), face);

		if (this->_gridType == DUAL)
		{
			this->SetBoundaryCopy(&this->_holder.at(this->_nextIndex), face);
		}
	}


	template<typename T>
	void Grid<T>::SetBoundaryCopy(Array<T>* arr, int face)
	{
		int ghostLayer = GetGhostLayer();
		for (int g = ghostLayer - 1; g >= 0; g--)
		{
			if (arr->GetDimCount() == DIM_1D)
			{
				int dimX = arr->GetDim(X);

				if (face == LEFT)
				{
					arr->Set(g, arr->Get(g + 1));
				}
				else if (face == RIGHT)
				{
					arr->Set(dimX - ghostLayer + g, arr->Get(dimX - ghostLayer + g - 1));
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
						arr->Set(g, j, arr->Get(g + 1, j));
					}
				}
				else if (face == RIGHT)
				{
					for (int j = 0; j < dimY; j++)
					{
						arr->Set(dimX - ghostLayer + g, j, arr->Get(dimX - ghostLayer + g - 1, j));
					}
				}
				else if (face == UP)
				{
					for (int i = 0; i < dimX; i++)
					{
						arr->Set(i, g, arr->Get(i, g + 1));
					}
				}
				else if (face == DOWN)
				{
					for (int i = 0; i < dimX; i++)
					{
						arr->Set(i, dimY - ghostLayer + g, arr->Get(i, dimY - ghostLayer + g - 1));
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
							arr->Set(g, j, k, arr->Get(g + 1, j, k));
						}
					}
				}
				else if (face == RIGHT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int j = 0; j < dimY; j++)
						{
							arr->Set(dimX - ghostLayer + g, j, k, arr->Get(dimX - ghostLayer + g - 1, j, k));
						}
					}
				}
				else if (face == FRONT)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, g, k, arr->Get(i, g + 1, k));
						}
					}
				}
				else if (face == BACK)
				{
					for (int k = 0; k < dimZ; k++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, dimY - ghostLayer + g, k, arr->Get(i, dimY - ghostLayer + g - 1, k));
						}
					}
				}
				else if (face == DOWN)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, j, g, arr->Get(i, j, g + 1));
						}
					}
				}
				else if (face == UP)
				{
					for (int j = 0; j < dimY; j++)
					{
						for (int i = 0; i < dimX; i++)
						{
							arr->Set(i, j, dimZ - ghostLayer + g, arr->Get(i, j, dimZ - ghostLayer + g - 1));
						}
					}
				}
			}
		}
	}


	template<typename T>
	void Grid<T>::Reset()
	{
		this->_nDim = 0;

		this->_full.Reset();

		if (this->_holder.size() > 0)
		{
			this->_holder.at(this->_currIndex).Reset();
			this->_holder.at(this->_nextIndex).Reset();
		}

		this->_haloBuffer.Reset();
		this->_gridType = SINGLE;
		this->_currIndex = 0;
		this->_nextIndex = 0;
	}


	template<typename T>
	void Grid<T>::Swap()
	{
		std::swap(this->_currIndex, this->_nextIndex);
	}

}


#endif