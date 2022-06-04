/**
 *
 * This is the class to process halo cells
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef HALOBUFFER_HPP
#define HALOBUFFER_HPP


#include <vector>
#include "Env.hpp"
#include "Common.hpp"


namespace APPFIS
{

	template<class T>
	class HaloBuffer
	{

	private:

		int _nFace = 0;
		int* _haloSizes = NULL;
		std::vector<T*> _sendBuffers;
		std::vector<T*> _recvBuffers;


	public:

		HaloBuffer<T>();
		~HaloBuffer<T>();

		HaloBuffer<T>(const HaloBuffer<T>& haloBuffer);
		HaloBuffer<T>& operator=(const HaloBuffer<T>& haloBuffer);

		void Set(int face, int index, T val);
		T Get(int face, int index) const;

		void Create(int nDim, int* dims);
		void Reset();

		int GetHaloSize(int face) const;
		int* GetHaloSizes();
		int GetFaceCount() const;

		T* GetSendBuffer(int face);
		T* GetRecvBuffer(int face);
	};


	template<class T>
	HaloBuffer<T>::HaloBuffer()
	{
	}


	template<class T>
	HaloBuffer<T>::~HaloBuffer()
	{
		this->Reset();
	}


	template<class T>
	HaloBuffer<T>::HaloBuffer(const HaloBuffer<T>& haloBuffer)
	{
		this->Reset();

		if (haloBuffer._nFace == 0)
		{
			return;
		}

		this->_nFace = haloBuffer._nFace;
		this->_haloSizes = new int[this->_nFace]();

		for (int i = 0; i < this->_nFace; i++)
		{
			this->_haloSizes[i] = haloBuffer._haloSizes[i];
		}

		if (this->_sendBuffers.size() == 0)
		{
			this->_sendBuffers = std::vector<T*>(this->_nFace);
			this->_recvBuffers = std::vector<T*>(this->_nFace);
		}

#ifdef APPFIS_CUDA
		for (int i = 0; i < this->_nFace; i++)
		{
			T* data;
			cudaMallocHost((void**)&data, this->_haloSizes[i] * sizeof(T));
			this->_sendBuffers[i] = data;

			T* data2;
			cudaMallocHost((void**)&data2, this->_haloSizes[i] * sizeof(T));
			this->_recvBuffers[i] = data2;
		}
#else
		T* data;
		for (int i = 0; i < this->_nFace; i++)
		{
			data = new T[this->_haloSizes[i]]();
			this->_sendBuffers[i] = data;

			data = new T[this->_haloSizes[i]]();
			this->_recvBuffers[i] = data;
		}
#endif

	}


	template<typename T>
	HaloBuffer<T>& HaloBuffer<T>::operator=(const HaloBuffer<T>& haloBuffer)
	{
		this->Reset();

		if (haloBuffer._nFace == 0)
		{
			return *this;
		}

		this->_nFace = haloBuffer._nFace;
		this->_haloSizes = new int[this->_nFace]();

		for (int i = 0; i < this->_nFace; i++)
		{
			this->_haloSizes[i] = haloBuffer._haloSizes[i];
		}

		if (this->_sendBuffers.size() == 0)
		{
			this->_sendBuffers = std::vector<T*>(this->_nFace);
			this->_recvBuffers = std::vector<T*>(this->_nFace);
		}

#ifdef APPFIS_CUDA
		for (int i = 0; i < this->_nFace; i++)
		{
			T* data;
			cudaMallocHost((void**)&data, this->_haloSizes[i] * sizeof(T));
			this->_sendBuffers[i] = data;

			T* data2;
			cudaMallocHost((void**)&data2, this->_haloSizes[i] * sizeof(T));
			this->_recvBuffers[i] = data2;
		}
#else
		T* data;
		for (int i = 0; i < this->_nFace; i++)
		{
			data = new T[this->_haloSizes[i]]();
			this->_sendBuffers[i] = data;

			data = new T[this->_haloSizes[i]]();
			this->_recvBuffers[i] = data;
		}
#endif

		return *this;
	}


	template<typename T>
	void HaloBuffer<T>::Set(int face, int index, T val)
	{
		this->_sendBuffers[face][index] = val;
	}


	template<typename T>
	T HaloBuffer<T>::Get(int face, int index) const
	{
		return this->_recvBuffers[face][index];
	}


	template<typename T>
	void HaloBuffer<T>::Create(int nDim, int* dims)
	{
		this->Reset();

		this->_nFace = 2 * nDim;
		this->_sendBuffers = std::vector<T*>(this->_nFace);
		this->_recvBuffers = std::vector<T*>(this->_nFace);
		this->_haloSizes = new int[this->_nFace]();

		if (this->_nFace == 2)
		{
			this->_haloSizes[LEFT] = 1;
			this->_haloSizes[RIGHT] = 1;
		}
		else if (this->_nFace == 4)
		{
			this->_haloSizes[LEFT] = dims[Y];
			this->_haloSizes[RIGHT] = dims[Y];
			this->_haloSizes[UP] = dims[X];
			this->_haloSizes[DOWN] = dims[X];
		}
		else if (this->_nFace == 6)
		{
			this->_haloSizes[LEFT] = dims[Y] * dims[Z];
			this->_haloSizes[RIGHT] = dims[Y] * dims[Z];
			this->_haloSizes[UP] = dims[X] * dims[Y];
			this->_haloSizes[DOWN] = dims[X] * dims[Y];
			this->_haloSizes[FRONT] = dims[X] * dims[Z];
			this->_haloSizes[BACK] = dims[X] * dims[Z];
		}
		else
		{
			std::cerr << "Dimension not supported" << std::endl;
			return;
		}

		int ghostLayer = GetGhostLayer();
		if (ghostLayer > 0)
		{
			for (int i = 0; i < this->_nFace; i++)
			{
				this->_haloSizes[i] *= ghostLayer;
			}
		}

#ifdef APPFIS_CUDA
		for (int i = 0; i < this->_nFace; i++)
		{
			T* data;
			cudaMallocHost((void**)&data, this->_haloSizes[i] * sizeof(T));
			this->_sendBuffers[i] = data;

			T* data2;
			cudaMallocHost((void**)&data2, this->_haloSizes[i] * sizeof(T));
			this->_recvBuffers[i] = data2;
		}
#else
		T* data;
		for (int i = 0; i < this->_nFace; i++)
		{
			data = new T[this->_haloSizes[i]]();
			this->_sendBuffers[i] = data;

			data = new T[this->_haloSizes[i]]();
			this->_recvBuffers[i] = data;
		}
#endif
	}


	template<typename T>
	void HaloBuffer<T>::Reset()
	{
		this->_nFace = 0;

		if (this->_haloSizes != NULL)
		{
			delete[] this->_haloSizes;
			this->_haloSizes = NULL;
		}

		while (!this->_sendBuffers.empty())
		{
			if (this->_sendBuffers.back() != NULL)
			{
#ifdef APPFIS_CUDA
				cudaFreeHost(this->_sendBuffers.back());
#else
				delete[] this->_sendBuffers.back();
#endif
			}
			this->_sendBuffers.pop_back();
		}

		while (!this->_recvBuffers.empty())
		{
			if (this->_recvBuffers.back() != NULL)
			{
#ifdef APPFIS_CUDA
				cudaFreeHost(this->_recvBuffers.back());
#else
				delete[] this->_recvBuffers.back();
#endif
			}
			this->_recvBuffers.pop_back();
		}
	}


	template<typename T>
	int HaloBuffer<T>::GetHaloSize(int face) const
	{
		return this->_haloSizes[face];
	}


	template<typename T>
	int* HaloBuffer<T>::GetHaloSizes()
	{
		return this->_haloSizes;
	}


	template<typename T>
	int HaloBuffer<T>::GetFaceCount() const
	{
		return this->_nFace;
	}


	template<typename T>
	T* HaloBuffer<T>::GetSendBuffer(int face)
	{
		return this->_sendBuffers[face];
	}


	template<typename T>
	T* HaloBuffer<T>::GetRecvBuffer(int face)
	{
		return this->_recvBuffers[face];
	}

}


#endif