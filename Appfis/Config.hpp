/**
 *
 * Structure to contain all kernel execution parameter
 * Author: Md Bulbul Sharif
 *
 **/


#include <vector>
#include "Grid.hpp"


#ifndef CONFIG_HPP
#define CONFIG_HPP


namespace APPFIS
{
	template<class T>
	class Config
	{

	private:
		int _dimX = 0;
		int _dimY = 0;
		int _dimZ = 0;
		int _startDimX = 0;
		int _endDimX = 0;
		int _startDimY = 0;
		int _endDimY = 0;
		int _startDimZ = 0;
		int _endDimZ = 0;

	public:
		std::vector<Grid<T>*> updatedGrids;

		Config<T>();
		~Config<T>();

		Config<T>(Grid<T>& g);
		Config<T>(Grid<T>& g, int skipStartX, int skipEndX);
		Config<T>(Grid<T>& g, int skipStartX, int skipEndX, int skipStartY, int skipEndY);
		Config<T>(Grid<T>& g, int skipStartX, int skipEndX, int skipStartY, int skipEndY, int skipStartZ, int skipEndZ);
		Config<T>(int dimX, int skipStartX, int skipEndX);
		Config<T>(int dimX, int skipStartX, int skipEndX, int dimY, int skipStartY, int skipEndY);
		Config<T>(int dimX, int skipStartX, int skipEndX, int dimY, int skipStartY, int skipEndY, int dimZ, int skipStartZ, int skipEndZ);

		int GetDimX() const;
		int GetDimY() const;
		int GetDimZ() const;
		int GetStartX() const;
		int GetStartY() const;
		int GetStartZ() const;
		int GetEndX() const;
		int GetEndY() const;
		int GetEndZ() const;

		bool IsUpdated();


		void GridsToUpdate(Grid<T>& grid)
		{
			updatedGrids.push_back(&grid);
		}


		template<typename... TArgs>
		void GridsToUpdate(Grid<T>& grid, TArgs&&... args)
		{
			updatedGrids.push_back(&grid);
			GridsToUpdate(args...);
		}

	};


	template<class T>
	Config<T>::Config()
	{
	}


	template<class T>
	Config<T>::~Config()
	{
	}


	template<class T>
	Config<T>::Config(Grid<T>& g)
	{
		Array<T>* arr = g.GetSubGridCPU();
		int nDim = arr->GetDimCount();
		int ghostLayer = GetGhostLayer();

		this->_dimX = arr->GetDim(X);
		this->_startDimX = ghostLayer;
		this->_endDimX = this->_dimX - ghostLayer;

		if (nDim > 1)
		{
			this->_dimY = arr->GetDim(Y);
			this->_startDimY = ghostLayer;
			this->_endDimY = this->_dimY - ghostLayer;
		}

		if (nDim > 2)
		{
			this->_dimZ = arr->GetDim(Z);
			this->_startDimZ = ghostLayer;
			this->_endDimZ = this->_dimZ - ghostLayer;
		}
	}


	template<class T>
	Config<T>::Config(Grid<T>& g, int skipStartX, int skipEndX)
	{
		Array<T>* arr = g.GetSubGridCPU();
		this->_dimX = arr->GetDim(X);
		this->_startDimX = skipStartX;
		this->_endDimX = this->_dimX - skipEndX;
	}


	template<class T>
	Config<T>::Config(Grid<T>& g, int skipStartX, int skipEndX, int skipStartY, int skipEndY)
	{
		Array<T>* arr = g.GetSubGridCPU();
		this->_dimX = arr->GetDim(X);
		this->_startDimX = skipStartX;
		this->_endDimX = this->_dimX - skipEndX;
		this->_dimY = arr->GetDim(Y);
		this->_startDimY = skipStartY;
		this->_endDimY = this->_dimY - skipEndY;
	}


	template<class T>
	Config<T>::Config(Grid<T>& g, int skipStartX, int skipEndX, int skipStartY, int skipEndY, int skipStartZ, int skipEndZ)
	{
		Array<T>* arr = g.GetSubGridCPU();
		this->_dimX = arr->GetDim(X);
		this->_startDimX = skipStartX;
		this->_endDimX = this->_dimX - skipEndX;
		this->_dimY = arr->GetDim(Y);
		this->_startDimY = skipStartY;
		this->_endDimY = this->_dimY - skipEndY;
		this->_dimZ = arr->GetDim(Z);
		this->_startDimZ = skipStartZ;
		this->_endDimZ = this->_dimZ - skipEndZ;
	}


	template<class T>
	Config<T>::Config(int dimX, int skipStartX, int skipEndX)
	{
		this->_dimX = dimX;
		this->_startDimX = skipStartX;
		this->_endDimX = this->_dimX - skipEndX;
	}


	template<class T>
	Config<T>::Config(int dimX, int skipStartX, int skipEndX, int dimY, int skipStartY, int skipEndY)
	{
		this->_dimX = dimX;
		this->_startDimX = skipStartX;
		this->_endDimX = this->_dimX - skipEndX;
		this->_dimY = dimY;
		this->_startDimY = skipStartY;
		this->_endDimY = this->_dimY - skipEndY;
	}


	template<class T>
	Config<T>::Config(int dimX, int skipStartX, int skipEndX, int dimY, int skipStartY, int skipEndY, int dimZ, int skipStartZ, int skipEndZ)
	{
		this->_dimX = dimX;
		this->_startDimX = skipStartX;
		this->_endDimX = this->_dimX - skipEndX;
		this->_dimY = dimY;
		this->_startDimY = skipStartY;
		this->_endDimY = this->_dimY - skipEndY;
		this->_dimZ = dimZ;
		this->_startDimZ = skipStartZ;
		this->_endDimZ = this->_dimZ - skipEndZ;
	}


	template<typename T>
	int Config<T>::GetDimX() const
	{
		return this->_dimX;
	}


	template<typename T>
	int Config<T>::GetStartX() const
	{
		return this->_startDimX;
	}


	template<typename T>
	int Config<T>::GetEndX() const
	{
		return this->_endDimX;
	}


	template<typename T>
	int Config<T>::GetDimY() const
	{
		return this->_dimY;
	}


	template<typename T>
	int Config<T>::GetStartY() const
	{
		return this->_startDimY;
	}


	template<typename T>
	int Config<T>::GetEndY() const
	{
		return this->_endDimY;
	}


	template<typename T>
	int Config<T>::GetDimZ() const
	{
		return this->_dimZ;
	}


	template<typename T>
	int Config<T>::GetStartZ() const
	{
		return this->_startDimZ;
	}


	template<typename T>
	int Config<T>::GetEndZ() const
	{
		return this->_endDimZ;
	}


	template<typename T>
	bool Config<T>::IsUpdated()
	{
		return static_cast<int>(this->updatedGrids.size()) > 0;
	}

}


#endif