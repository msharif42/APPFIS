/**
 *
 * This is the custom data object class.
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef ARRAY_HPP
#define ARRAY_HPP


#include <fstream>
#include "Env.hpp"
#include "Utils.hpp"


namespace APPFIS
{

	template<class T>
	class Array
	{

	private:
		int _nDim = 0;
		int _size = 0;
		int* _dims = NULL;
		T* _data = NULL;

	public:
		Array<T>();
		~Array<T>();

		Array<T>(int nDim, int dimX);
		Array<T>(int nDim, int dimX, int dimY);
		Array<T>(int nDim, int dimX, int dimY, int dimZ);
		Array<T>(int nDim, int* dims);

		Array<T>(const Array<T>& arr);
		Array& operator=(const Array<T>& arr);

		EXEC_SPACE void Set(int i, T val);
		EXEC_SPACE void Set(int i, int j, T val);
		EXEC_SPACE void Set(int i, int j, int k, T val);
		EXEC_SPACE T Get(int i);
		EXEC_SPACE T Get(int i, int j);
		EXEC_SPACE T Get(int i, int j, int k);

		int GetDimCount() const;
		int GetSize() const;
		int GetDim(int dim) const;
		int* GetDims() const;

		T* GetData();
		void SetData(T* data);
		void FillData(T value);
		void Square();

		void Reset();
		void AllocateMemory();
		void Resize(int nDim, int* dims);

		void LoadAsciiFile(const std::string& filePath, int headerSize = 0);
		void LoadBinaryFile(const std::string& filePath, int headerSize = 0);
		void SaveAsciiFile(const std::string& filePath);
		void SaveBinaryFile(const std::string& filePath);

		std::string ToString();

		T** DataAddress();
		int** DimsAddress();

	private:
		EXEC_SPACE int Get1DIndex(int i, int j);
		EXEC_SPACE int Get1DIndex(int i, int j, int k);

	};


	template<class T>
	Array<T>::Array()
	{
	}


	template<class T>
	Array<T>::~Array()
	{
		this->Reset();
	}


	template<class T>
	Array<T>::Array(int nDim, int dimX)
	{
		this->_nDim = nDim;
		this->_dims = new int[this->_nDim]();
		this->_dims[X] = dimX;
		this->_size = dimX;
	}


	template<class T>
	Array<T>::Array(int nDim, int dimX, int dimY)
	{
		this->_nDim = nDim;
		this->_dims = new int[this->_nDim]();
		this->_dims[X] = dimX;
		this->_dims[Y] = dimY;
		this->_size = dimX * dimY;
	}


	template<class T>
	Array<T>::Array(int nDim, int dimX, int dimY, int dimZ)
	{
		this->_nDim = nDim;
		this->_dims = new int[this->_nDim]();
		this->_dims[X] = dimX;
		this->_dims[Y] = dimY;
		this->_dims[Z] = dimZ;
		this->_size = dimX * dimY * dimZ;
	}


	template<class T>
	Array<T>::Array(int nDim, int* dims)
	{
		this->_nDim = nDim;
		this->_dims = new int[this->_nDim]();
		this->_size = 1;

		for (int i = 0; i < this->_nDim; i++)
		{
			this->_dims[i] = dims[i];
			this->_size *= dims[i];
		}
	}


	template<class T>
	Array<T>::Array(const Array<T>& arr)
	{
		std::cout << "Array copy" << std::endl;

		this->Reset();

		if (arr._nDim == 0) return;

		this->_nDim = arr._nDim;
		this->_dims = new int[this->_nDim]();
		this->_size = arr._size;

		for (int i = 0; i < this->_nDim; i++)
		{
			this->_dims[i] = arr._dims[i];
		}

		if (arr._data != NULL)
		{
			this->AllocateMemory();
			this->SetData(arr._data);
		}
	}


	template<typename T>
	Array<T>& Array<T>::operator=(const Array<T>& arr)
	{
		std::cout << "Array assign" << std::endl;

		this->Reset();

		if (arr._nDim == 0) return *this;

		this->_nDim = arr._nDim;
		this->_dims = new int[this->_nDim]();
		this->_size = arr._size;

		for (int i = 0; i < this->_nDim; i++)
		{
			this->_dims[i] = arr._dims[i];
		}

		if (arr._data != NULL)
		{
			this->AllocateMemory();
			this->SetData(arr._data);
		}

		return *this;
	}


	template<typename T>
	EXEC_SPACE void Array<T>::Set(int i, T val)
	{
		this->_data[i] = val;
	}


	template<typename T>
	EXEC_SPACE void Array<T>::Set(int i, int j, T val)
	{
		this->_data[this->Get1DIndex(i, j)] = val;
	}


	template<typename T>
	EXEC_SPACE void Array<T>::Set(int i, int j, int k, T val)
	{
		this->_data[this->Get1DIndex(i, j, k)] = val;
	}


	template<typename T>
	EXEC_SPACE T Array<T>::Get(int i)
	{
		return this->_data[i];
	}


	template<typename T>
	EXEC_SPACE T Array<T>::Get(int i, int j)
	{
		return this->_data[this->Get1DIndex(i, j)];
	}


	template<typename T>
	EXEC_SPACE T Array<T>::Get(int i, int j, int k)
	{
		return this->_data[this->Get1DIndex(i, j, k)];
	}


	template<typename T>
	int Array<T>::GetDimCount() const
	{
		return this->_nDim;
	}


	template<typename T>
	int Array<T>::GetSize() const
	{
		return this->_size;
	}


	template<typename T>
	int Array<T>::GetDim(int dim) const
	{
		return this->_dims[dim];
	}


	template<typename T>
	int* Array<T>::GetDims() const
	{
		return this->_dims;
	}


	template<typename T>
	T* Array<T>::GetData()
	{
		return this->_data;
	}


	template<typename T>
	void Array<T>::SetData(T* data)
	{
		if (this->_data == NULL && this->_size > 0)
		{
			this->AllocateMemory();
		}

		for (int i = 0; i < this->_size; i++)
		{
			this->_data[i] = data[i];
		}
	}


	template<typename T>
	void Array<T>::Square()
	{
		if (this->_data == NULL && this->_size > 0)
		{
			this->AllocateMemory();
		}

		for (int i = 0; i < this->_size; i++)
		{
			this->_data[i] = this->_data[i] * this->_data[i];
		}
	}


	template<typename T>
	void Array<T>::FillData(T value)
	{
		if (this->_data == NULL && this->_size > 0)
		{
			this->AllocateMemory();
		}

		for (int i = 0; i < this->_size; i++)
		{
			this->_data[i] = value;
		}
	}


	template<typename T>
	void Array<T>::Reset()
	{
		if (this->_data != NULL)
		{
			delete[] this->_data;
			this->_data = NULL;
		}

		if (this->_dims != NULL)
		{
			delete[] this->_dims;
			this->_dims = NULL;
		}

		this->_nDim = 0;
		this->_size = 0;
	}


	template<typename T>
	void Array<T>::AllocateMemory()
	{
		this->_data = new T[this->_size]();
	}


	template<typename T>
	T** Array<T>::DataAddress()
	{
		return &this->_data;
	}


	template<typename T>
	int** Array<T>::DimsAddress()
	{
		return &this->_dims;
	}


	template<typename T>
	void Array<T>::Resize(int nDim, int* dims)
	{
		this->Reset();

		this->_nDim = nDim;
		this->_dims = new int[this->_nDim]();
		this->_size = 1;

		for (int i = 0; i < this->_nDim; i++)
		{
			this->_dims[i] = dims[i];
			this->_size *= dims[i];
		}
	}


	template<typename T>
	void Array<T>::LoadAsciiFile(const std::string& filePath, int headerSize)
	{
		std::ifstream inFile(filePath.c_str());

		if (!inFile.is_open())
		{
			std::cerr << "Can not read " << filePath << std::endl;
			return;
		}

		if (this->_data == NULL && this->_size > 0)
		{
			this->AllocateMemory();
		}

		int index = 0;
		int lineNum = 0;
		std::string line;

		std::getline(inFile, line);

		while (!inFile.eof())
		{
			lineNum++;
			if (lineNum > headerSize)
			{
				std::vector<std::string> lineVals = Split(line, ' ');

				std::string val;
				std::vector<std::string>::iterator strit = lineVals.begin();

				for (; strit != lineVals.end() && index < this->_size; strit++, index++)
				{
					val = *strit;
					this->_data[index] = static_cast<T>(atof(val.c_str()));
				}
			}
			std::getline(inFile, line);
		}
		inFile.close();
	}


	template<typename T>
	void Array<T>::LoadBinaryFile(const std::string& filePath, int headerSize)
	{
		std::ifstream inFile(filePath.c_str(), std::ios::binary);

		if (!inFile.is_open())
		{
			std::cerr << "Can not read " << filePath << std::endl;
			return;
		}

		if (this->_data == NULL && this->_size > 0)
		{
			this->AllocateMemory();
		}

		if (headerSize > 0)
		{
			inFile.seekg(headerSize * sizeof(T), std::ios::beg);
		}
		inFile.read((char*)this->_data, sizeof(T) * this->_size);

		inFile.close();
	}


	template<typename T>
	void Array<T>::SaveAsciiFile(const std::string& filePath)
	{
		CreateDir(GetParentDir(filePath.c_str()));

		std::ofstream outFile(filePath.c_str());

		if (!outFile.is_open() || this->_data == NULL)
		{
			std::cerr << "Can not write to " << filePath << std::endl;
			return;
		}

		outFile.precision(6);
		outFile << std::fixed;

		if (this->_nDim == DIM_1D)
		{
			for (int i = 0; i < this->GetDim(X); i++)
			{
				outFile << this->Get(i);
				if (i < this->GetDim(X) - 1)
				{
					outFile << " ";
				}
			}
		}
		else if (this->_nDim == DIM_2D)
		{
			for (int j = 0; j < this->GetDim(Y); j++)
			{
				for (int i = 0; i < this->GetDim(X); i++)
				{
					outFile << this->Get(i, j);
					if (i < this->GetDim(X) - 1)
					{
						outFile << " ";
					}
				}
				outFile << "\n";
			}
		}
		else if (this->_nDim == DIM_3D)
		{
			for (int k = 0; k < this->GetDim(Z); k++)
			{
				for (int j = 0; j < this->GetDim(Y); j++)
				{
					for (int i = 0; i < this->GetDim(X); i++)
					{
						outFile << this->Get(i, j, k);
						if (i < this->GetDim(X) - 1)
						{
							outFile << " ";
						}
					}
					outFile << "\n";
				}
			}
		}

		outFile.close();
	}


	template<typename T>
	void Array<T>::SaveBinaryFile(const std::string& filePath)
	{
		CreateDir(GetParentDir(filePath.c_str()));

		std::ofstream outFile(filePath.c_str(), std::ios::binary);

		if (!outFile.is_open() || this->_data == NULL)
		{
			std::cerr << "Can not write to " << filePath << std::endl;
			return;
		}

		outFile.write((char*)&this->_data[0], this->_size * sizeof(T));

		outFile.close();
	}


	template<typename T>
	std::string Array<T>::ToString()
	{
		std::string s = "";

		if (this->_data == NULL) return s;

		if (this->_nDim == DIM_1D)
		{
			for (int i = 0; i < this->GetDim(X); i++)
			{
				s = s + " " + std::to_string(this->Get(i));
			}
		}
		else if (this->_nDim == DIM_2D)
		{
			for (int j = 0; j < this->GetDim(Y); j++)
			{
				for (int i = 0; i < this->GetDim(X); i++)
				{
					s = s + " " + std::to_string(this->Get(i, j));
				}
				s = s + "\n";
			}
		}
		else if (this->_nDim == DIM_3D)
		{
			bool flag = false;
			for (int k = 0; k < this->GetDim(Z); k++)
			{
				for (int j = 0; j < this->GetDim(Y); j++)
				{
					if (flag)
						s = s + "    ";
					for (int i = 0; i < this->GetDim(X); i++)
					{
						s = s + " " + std::to_string(this->Get(i, j, k));
					}
					s = s + "\n";
				}
				flag = !flag;
			}
		}

		return s;
	}


	template<typename T>
	EXEC_SPACE int Array<T>::Get1DIndex(int i, int j)
	{
		return  i + this->_dims[X] * j;
	}


	template<typename T>
	EXEC_SPACE int Array<T>::Get1DIndex(int i, int j, int k)
	{
		return  i + this->_dims[X] * j + this->_dims[X] * this->_dims[Y] * k;
	}

}


#endif