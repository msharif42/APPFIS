/**
 *
 * This is the singleton class that handles process topology.
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef TOPOLOGY_HPP
#define TOPOLOGY_HPP


#include <omp.h>
#include "mpi.h"
#include "Env.hpp"
#include "Attribute.hpp"
#include <string>
#include <limits> 

#ifdef APPFIS_CUDA
#include <cuda_runtime.h>
#endif


namespace APPFIS
{

	class Topology
	{

	public:
		int _rank = 0;
		int _master = 0;
		int _nRank = 0;
		int _nNeighbour = 0;

		int _nDim = 0;
		int _pDim = 0;
		int _ghostLayer = 1;

		bool _overlap = false;
		bool _periodic = false;

		int* _blocks = NULL;
		int* _periods = NULL;
		int* _coords = NULL;
		int* _neighboursRanks = NULL;

		MPI_Comm _comm = NULL;

#ifdef APPFIS_CUDA
		cudaStream_t _streams;
#endif


	public:
		~Topology();

		static Topology& GetInstance()
		{
			static Topology instance;
			return instance;
		}

		Topology(Topology const&) = delete;
		void operator=(Topology const&) = delete;

		void Initialize(int argc, char* argv[], int dataDim, int partitionDim, Attribute attribute = ATTRIBUTE);
		void Finalize();


	private:
		Topology() {};
		std::string _neighboursNames[6] = { "Left", "Right", "Up", "Down", "Front", "Back" };

	};


	Topology::~Topology()
	{
		if (this->_blocks != NULL)
		{
			delete[] this->_blocks;
			this->_blocks = NULL;
		}
		if (this->_periods != NULL)
		{
			delete[] this->_periods;
			this->_periods = NULL;
		}
		if (this->_coords != NULL)
		{
			delete[] this->_coords;
			this->_coords = NULL;
		}
		if (this->_neighboursRanks != NULL)
		{
			delete[] this->_neighboursRanks;
			this->_neighboursRanks = NULL;
		}
	}


	void Topology::Initialize(int argc, char* argv[], int dataDim, int partitionDim, Attribute attribute)
	{
		if (partitionDim > dataDim)
		{
			std::cerr << "Initialization failed. Partition dimension (" << partitionDim << ") can not be larger than data dimension (" << dataDim << ")." << std::endl;
			exit(EXIT_FAILURE);
		}

		this->_nDim = dataDim;
		this->_pDim = partitionDim;
		this->_ghostLayer = attribute.ghostLayer;
		this->_overlap = attribute.overlap;
		this->_periodic = attribute.periodic;

		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &this->_nRank);
		MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);

		omp_set_num_threads(attribute.threads);

#ifdef APPFIS_CUDA
		int deviceId = 0;
		cudaError_t err = cudaGetDevice(&deviceId);
		if (err != cudaSuccess)
		{
			std::cerr << cudaGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		int deviceCount = 0;
		err = cudaGetDeviceCount(&deviceCount);
		if (err != cudaSuccess)
		{
			std::cerr << cudaGetErrorString(err) << std::endl;
			exit(EXIT_FAILURE);
		}

		if (deviceId != this->_rank % deviceCount)
		{
			err = cudaSetDevice(this->_rank % deviceCount);
			if (err != cudaSuccess)
			{
				std::cerr << cudaGetErrorString(err) << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		cudaStreamCreate(&_streams);
#endif

		this->_blocks = new int[this->_nDim]();
		this->_periods = new int[this->_nDim]();
		this->_coords = new int[this->_nDim]();

		this->_nNeighbour = this->_nDim * 2;
		this->_neighboursRanks = new int[this->_nNeighbour]();

		for (int i = 0; i < this->_nDim; i++)
		{
			this->_periods[i] = attribute.periodic;
		}

		for (int i = 0; i < this->_nDim - this->_pDim; i++)
		{
			this->_blocks[this->_nDim - 1 - i] = 1;
		}

		MPI_Dims_create(this->_nRank, this->_nDim, this->_blocks);

		int tempBlocks = this->_blocks[0];
		this->_blocks[0] = this->_blocks[this->_nDim - 1];
		this->_blocks[this->_nDim - 1] = tempBlocks;

		if (this->_rank == this->_master)
		{
			std::string s = "Process Dims:";
			for (int i = 0; i < this->_nDim; i++)
			{
				s = s + " " + std::to_string(this->_blocks[i]);
			}
			std::cout << s << std::endl;
		}

		MPI_Cart_create(MPI_COMM_WORLD, this->_nDim, this->_blocks, this->_periods, 1, &this->_comm);

		MPI_Cart_coords(this->_comm, this->_rank, this->_nDim, this->_coords);

		/*std::string s = "Rank: " + std::to_string(this->_rank) + " Pos:";
		for (int i = 0; i < this->_nDim; i++)
		{
			s = s + " " + std::to_string(this->_coords[i]);
		}
		std::cout << s << std::endl;*/

		MPI_Cart_shift(this->_comm, X, 1, &this->_neighboursRanks[LEFT], &this->_neighboursRanks[RIGHT]);
		if (this->_nDim == DIM_2D)
		{
			MPI_Cart_shift(this->_comm, Y, 1, &this->_neighboursRanks[UP], &this->_neighboursRanks[DOWN]);
		}
		else if (this->_nDim == DIM_3D)
		{
			MPI_Cart_shift(this->_comm, Y, 1, &this->_neighboursRanks[FRONT], &this->_neighboursRanks[BACK]);
			MPI_Cart_shift(this->_comm, Z, 1, &this->_neighboursRanks[DOWN], &this->_neighboursRanks[UP]);
		}

		MPI_Comm_rank(this->_comm, &this->_rank);

		/*for(int i = 0; i < this->_nNeighbour; i++)
		{
			if(this->_neighboursRanks[i] == MPI_PROC_NULL)
			{
				std::cout << "Process: " << this->_rank << ", I have no " << this->_neighboursNames[i] << " neighbour" << std::endl;
			}
			else
			{
				std::cout << "Process: " << this->_rank << ", I have a " << this->_neighboursNames[i] << " neighbour - Process " << this->_neighboursRanks[i] << std::endl;
			}
		}*/
	}


	void Topology::Finalize()
	{
		MPI_Finalize();
	}

}


#endif