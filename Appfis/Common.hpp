/**
 *
 * This is the file containing methods that can be used directly
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef COMMON_HPP
#define COMMON_HPP


#include "Topology.hpp"


namespace APPFIS
{

	void Initialize(int argc, char* argv[], int dataDim, int partitionDim, Attribute attribute = ATTRIBUTE);
	void Finalize();

	int GetRank();
	int GetMasterRank();
	int GetRankCount();
	int GetNeighbourCount();

	int GetDimCount();
	int GetPartitionDim();
	int GetGhostLayer();

	bool IsMaster();
	bool IsOverlap();
	bool IsPeriodic();

	int* GetBlocks();
	int* GetCoords();
	int GetNeighboursRank(int face);

	MPI_Comm& GetCommunicator();

#ifdef APPFIS_CUDA
	cudaStream_t& GetCudaStream();
#endif

	void SetMasterRank(int master);
	void GetLocalPartitionDims(int nDim, int* fullDataDims, int* subDataDims);
	void GetLocalPartitionDims(int nDim, int* coords, int* fullDataDims, int* subDataDims);


	void Initialize(int argc, char* argv[], int dataDim, int partitionDim, Attribute attribute)
	{
		Topology& topology = Topology::GetInstance();
		topology.Initialize(argc, argv, dataDim, partitionDim, attribute);
	}


	void Finalize()
	{
		Topology& topology = Topology::GetInstance();
		topology.Finalize();
	}


	int GetRank()
	{
		Topology& topology = Topology::GetInstance();
		return topology._rank;
	}


	int GetMasterRank()
	{
		Topology& topology = Topology::GetInstance();
		return topology._master;
	}


	int GetRankCount()
	{
		Topology& topology = Topology::GetInstance();
		return topology._nRank;
	}


	int GetNeighbourCount()
	{
		Topology& topology = Topology::GetInstance();
		return topology._nNeighbour;
	}


	int GetDimCount()
	{
		Topology& topology = Topology::GetInstance();
		return topology._nDim;
	}


	int GetPartitionDim()
	{
		Topology& topology = Topology::GetInstance();
		return topology._pDim;
	}


	int GetGhostLayer()
	{
		Topology& topology = Topology::GetInstance();
		return topology._ghostLayer;
	}


	bool IsMaster()
	{
		Topology& topology = Topology::GetInstance();
		return (topology._rank == topology._master);
	}


	bool IsOverlap()
	{
		Topology& topology = Topology::GetInstance();
		return topology._overlap;
	}


	bool IsPeriodic()
	{
		Topology& topology = Topology::GetInstance();
		return topology._periodic;
	}


	int* GetBlocks()
	{
		Topology& topology = Topology::GetInstance();
		return topology._blocks;
	}


	int* GetCoords()
	{
		Topology& topology = Topology::GetInstance();
		return topology._coords;
	}


	int GetNeighboursRank(int face)
	{
		Topology& topology = Topology::GetInstance();
		return topology._neighboursRanks[face];
	}


	MPI_Comm& GetCommunicator()
	{
		Topology& topology = Topology::GetInstance();
		return topology._comm;
	}


#ifdef APPFIS_CUDA
	cudaStream_t& GetCudaStream()
	{
		Topology& topology = Topology::GetInstance();
		return topology._streams;
	}
#endif


	void SetMasterRank(int master)
	{
		Topology& topology = Topology::GetInstance();
		if (master < 0 || master >= topology._nRank)
		{
			std::cerr << "Setting master rank to " << master << " has failed. Default value 0." << std::endl;
		}
		else
		{
			topology._master = master;
		}
	}


	void GetLocalPartitionDims(int nDim, int* fullDataDims, int* subDataDims)
	{
		if (nDim != GetDimCount())
		{
			std::cerr << "Dimension not matched." << std::endl;
			return;
		}

		GetLocalPartitionDims(nDim, GetCoords(), fullDataDims, subDataDims);
	}


	void GetLocalPartitionDims(int nDim, int* coords, int* fullDataDims, int* subDataDims)
	{
		int* blocks = GetBlocks();

		for (int i = 0; i < nDim; i++)
		{
			if (fullDataDims[i] < blocks[i])
			{
				std::cerr << "Number of process in a dimension: " << blocks[i] << " is larger than data dimension: " << fullDataDims[i] << std::endl;
				exit(EXIT_FAILURE);
			}
			if (fullDataDims[i] / blocks[i] < 2 * GetGhostLayer())
			{
				std::cerr << "Subgrid Data dimension should be at least: " << (2 * GetGhostLayer()) << ". Found: " << (fullDataDims[i] / blocks[i]) << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		for (int i = 0; i < nDim; i++)
		{
			subDataDims[i] = 2 * GetGhostLayer() + fullDataDims[i] / blocks[i];

			int rem = fullDataDims[i] % blocks[i];

			if (coords[i] < rem && rem > 0)
			{
				subDataDims[i]++;
			}
		}
	}

}


#endif