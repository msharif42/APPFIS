/**
 *
 * This is the file containing methods that can be used for inter process communication
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef COMMUNICATION_HPP
#define COMMUNICATION_HPP


#include "mpi.h"
#include "Env.hpp"
#include "Common.hpp"
#include "Array.hpp"
#include "HaloBuffer.hpp"
#include "Runtime.hpp"


namespace APPFIS
{

	template<typename T>
	void Scatter(Array<T>* global, Array<T>* local);

	template<typename T>
	void Gather(Array<T>* global, Array<T>* local);

	template<typename T>
	void Exchange(Array<T>* arr, HaloBuffer<T>* buffer);

	template<typename T>
	T AllReduce(T local, REDUCTION rType);

#ifdef APPFIS_CUDA
	template<typename T>
	void ExchangeGPU(Array<T>* arr, HaloBuffer<T>* buffer);
#endif

	template<typename T>
	void ExchangeInitial(Array<T>* arr, HaloBuffer<T>* buffer);

	int GetBlockTypeIndex(int nDim, int* dimValsLocal, int* typesVals, int* blockSizeVals);

	void ProcessSync();


	template<typename T>
	void Scatter(Array<T>* global, Array<T>* local)
	{
		if (global->GetDimCount() != GetDimCount())
		{
			std::cerr << "Partition failed. Data dimension and partition are different." << std::endl;
			exit(EXIT_FAILURE);
		}

		MPI_Comm& comm = GetCommunicator();

		int nDim = global->GetDimCount();
		int ghostLayer = GetGhostLayer();

		int* dimsGlobal = global->GetDims();
		int* dimsLocal = local->GetDims();

		MPI_Datatype dataType;
		if (std::is_same<T, int>::value)
		{
			dataType = MPI_INT;
		}
		else if (std::is_same<T, double>::value)
		{
			dataType = MPI_DOUBLE;
		}
		else if (std::is_same<T, float>::value)
		{
			dataType = MPI_FLOAT;
		}
		else
		{
			std::cerr << "Partition failed. Not supported datatype" << std::endl;
			exit(EXIT_FAILURE);
		}

		int recvCounts = 1;
		int recvDispls = 0;
		MPI_Datatype recvTypes;

		int* dimsLocalReverse = new int[nDim];
		for (int i = 0; i < nDim; i++)
		{
			dimsLocalReverse[i] = dimsLocal[nDim - 1 - i];
		}
		int* dimsLocalReverse2 = new int[nDim];
		for (int i = 0; i < nDim; i++)
		{
			dimsLocalReverse2[i] = dimsLocalReverse[i] - 2 * ghostLayer;
		}

		int* startsR = new int[nDim];
		for (int i = 0; i < nDim; i++)
		{
			startsR[i] = ghostLayer;
		}
		MPI_Type_create_subarray(nDim, dimsLocalReverse, dimsLocalReverse2, startsR, MPI_ORDER_C, dataType, &recvTypes);
		MPI_Type_commit(&recvTypes);

		if (IsMaster())
		{
			int nRank = GetRankCount();
			int* sendCounts = new int[nRank];
			int* sendDisplsT = new int[nRank];
			MPI_Datatype* sendTypes = new MPI_Datatype[nRank];

			int* typesVals = new int[nDim];
			int* blockSizeVals = new int[nDim];
			int* typeValsMaxIndex = new int[nDim];

			int* blocks = GetBlocks();
			for (int i = 0; i < nDim; i++)
			{
				blockSizeVals[i] = dimsGlobal[i] / blocks[i];
				if (dimsGlobal[i] % blocks[i] == 0)
				{
					typesVals[i] = 1;
				}
				else
				{
					typesVals[i] = 2;
				}

				typeValsMaxIndex[i] = dimsGlobal[i] % blocks[i];
			}

			int typesSize = 1;
			for (int i = 0; i < nDim; i++)
			{
				typesSize = typesSize * typesVals[i];
			}

			MPI_Datatype* blockTypes = new MPI_Datatype[typesSize];

			int* dimsGlobalReverse = new int[nDim];
			for (int i = 0; i < nDim; i++)
			{
				dimsGlobalReverse[i] = dimsGlobal[nDim - 1 - i];
			}

			int* subSizes = new int[nDim];
			int* starts = new int[nDim];
			for (int i = 0; i < nDim; i++)
			{
				starts[i] = 0;
			}

			if (nDim == DIM_1D)
			{
				for (int i = 0; i < typesVals[X]; i++)
				{
					subSizes[X] = blockSizeVals[X] + i;
					MPI_Type_create_subarray(nDim, dimsGlobalReverse, subSizes, starts, MPI_ORDER_C, dataType, &blockTypes[i]);
					MPI_Type_commit(&blockTypes[i]);
				}
			}
			else if (nDim == DIM_2D)
			{
				for (int j = 0; j < typesVals[Y]; j++)
				{
					subSizes[X] = blockSizeVals[Y] + j;
					for (int i = 0; i < typesVals[X]; i++)
					{
						subSizes[Y] = blockSizeVals[X] + i;

						MPI_Type_create_subarray(nDim, dimsGlobalReverse, subSizes, starts, MPI_ORDER_C, dataType, &blockTypes[i + typesVals[X] * j]);
						MPI_Type_commit(&blockTypes[i + typesVals[X] * j]);
					}
				}
			}
			else if (nDim == DIM_3D)
			{
				for (int k = 0; k < typesVals[Z]; k++)
				{
					subSizes[X] = blockSizeVals[Z] + k;
					for (int j = 0; j < typesVals[Y]; j++)
					{
						subSizes[Y] = blockSizeVals[Y] + j;
						for (int i = 0; i < typesVals[X]; i++)
						{
							subSizes[Z] = blockSizeVals[X] + i;

							MPI_Type_create_subarray(nDim, dimsGlobalReverse, subSizes, starts, MPI_ORDER_C, dataType, &blockTypes[i + typesVals[X] * j + typesVals[X] * typesVals[Y] * k]);
							MPI_Type_commit(&blockTypes[i + typesVals[X] * j + typesVals[X] * typesVals[Y] * k]);
						}
					}
				}
			}

			for (int proc = 0; proc < nRank; proc++)
			{
				int* coords = new int[nDim];
				int* dimValsLocal = new int[nDim];
				int* dimValsLocalT = new int[nDim];

				MPI_Cart_coords(comm, proc, nDim, coords);
				GetLocalPartitionDims(nDim, coords, dimsGlobal, dimValsLocalT);

				for (int i = 0; i < nDim; i++)
				{
					dimValsLocal[i] = dimValsLocalT[i] - 2 * ghostLayer;
				}

				int displsIndex = 0;

				int displsXIndex = 0;
				if (typeValsMaxIndex[X] == 0 || coords[X] < typeValsMaxIndex[X])
				{
					displsXIndex += (coords[X] * dimValsLocal[X]);
				}
				else
				{
					displsXIndex += ((typeValsMaxIndex[X]) * (dimValsLocal[X] + 1) + (coords[X] - typeValsMaxIndex[X]) * (dimValsLocal[X]));
				}

				displsIndex = displsXIndex;

				if (nDim == DIM_2D || nDim == DIM_3D)
				{
					int displsYIndex = 0;
					if (typeValsMaxIndex[Y] == 0 || coords[Y] < typeValsMaxIndex[Y])
					{
						displsYIndex += (coords[Y] * dimValsLocal[Y]);
					}
					else
					{
						displsYIndex += ((typeValsMaxIndex[Y]) * (dimValsLocal[Y] + 1) + (coords[Y] - typeValsMaxIndex[Y]) * (dimValsLocal[Y]));
					}

					displsIndex += dimsGlobal[X] * displsYIndex;
				}

				if (nDim == DIM_3D)
				{
					int displsZIndex = 0;
					if (typeValsMaxIndex[Z] == 0 || coords[Z] < typeValsMaxIndex[Z])
					{
						displsZIndex += (coords[Z] * dimValsLocal[Z]);
					}
					else
					{
						displsZIndex += ((typeValsMaxIndex[Z]) * (dimValsLocal[Z] + 1) + (coords[Z] - typeValsMaxIndex[Z]) * (dimValsLocal[Z]));
					}

					displsIndex += dimsGlobal[X] * dimsGlobal[Y] * displsZIndex;
				}

				sendCounts[proc] = 1;
				sendDisplsT[proc] = displsIndex;
				sendTypes[proc] = blockTypes[GetBlockTypeIndex(nDim, dimValsLocal, typesVals, blockSizeVals)];

				delete[] coords;
				delete[] dimValsLocal;
				delete[] dimValsLocalT;
			}

			MPI_Status* status = new MPI_Status[nRank];
			MPI_Request* request = new MPI_Request[nRank];
			T* globalData = global->GetData();

			for (int proc = 0; proc < nRank; proc++)
			{
				if (proc == GetMasterRank())
				{
					T* localData = local->GetData();
					MPI_Sendrecv(&(globalData[sendDisplsT[proc]]), sendCounts[proc], sendTypes[proc], proc, 0,
						&(localData[recvDispls]), recvCounts, recvTypes, proc, 0,
						comm, &status[proc]);
				}
				else
				{
					MPI_Isend(&(globalData[sendDisplsT[proc]]), sendCounts[proc], sendTypes[proc], proc, 0, comm, &request[proc]);
				}
			}

			for (int proc = 1; proc < nRank; proc++)
			{
				MPI_Wait(&request[proc], &status[proc]);
			}

			for (int i = 0; i < typesSize; i++)
			{
				MPI_Type_free(&blockTypes[i]);
			}

			delete[] sendCounts;
			delete[] sendDisplsT;
			delete[] sendTypes;

			delete[] typesVals;
			delete[] blockSizeVals;
			delete[] typeValsMaxIndex;

			delete[] blockTypes;
			delete[] dimsGlobalReverse;

			delete[] subSizes;
			delete[] starts;

			delete[] status;
			delete[] request;
		}
		else
		{
			MPI_Status recvStatus;
			MPI_Request recvRequest;
			T* localData = local->GetData();
			MPI_Irecv(&(localData[recvDispls]), recvCounts, recvTypes, GetMasterRank(), 0, comm, &recvRequest);

			MPI_Wait(&recvRequest, &recvStatus);
		}

		MPI_Type_free(&recvTypes);

		MPI_Barrier(comm);

		delete[] dimsLocalReverse;
		delete[] dimsLocalReverse2;
		delete[] startsR;
	}


	template<typename T>
	void Gather(Array<T>* global, Array<T>* local)
	{
		if (global->GetDimCount() != GetDimCount())
		{
			std::cerr << "Partition failed. Data dimension and partition are different." << std::endl;
			exit(EXIT_FAILURE);
		}

		MPI_Comm& comm = GetCommunicator();

		int nDim = global->GetDimCount();
		int ghostLayer = GetGhostLayer();

		int* dimsGlobal = global->GetDims();
		int* dimsLocal = local->GetDims();

		MPI_Datatype dataType;
		if (std::is_same<T, int>::value)
		{
			dataType = MPI_INT;
		}
		else if (std::is_same<T, double>::value)
		{
			dataType = MPI_DOUBLE;
		}
		else if (std::is_same<T, float>::value)
		{
			dataType = MPI_FLOAT;
		}
		else
		{
			std::cerr << "Partition failed. Not supported MPI datatype" << std::endl;
			exit(EXIT_FAILURE);
		}

		int sendCounts = 1;
		int sendDispls = 0;
		MPI_Datatype sendTypes;

		int* dimsLocalReverse = new int[nDim];
		for (int i = 0; i < nDim; i++)
		{
			dimsLocalReverse[i] = dimsLocal[nDim - 1 - i];
		}
		int* dimsLocalReverse2 = new int[nDim];
		for (int i = 0; i < nDim; i++)
		{
			dimsLocalReverse2[i] = dimsLocalReverse[i] - 2 * ghostLayer;
		}

		int* startsS = new int[nDim];
		for (int i = 0; i < nDim; i++)
		{
			startsS[i] = ghostLayer;
		}
		MPI_Type_create_subarray(nDim, dimsLocalReverse, dimsLocalReverse2, startsS, MPI_ORDER_C, dataType, &sendTypes);
		MPI_Type_commit(&sendTypes);

		if (IsMaster())
		{
			int nRank = GetRankCount();
			int* recvCounts = new int[nRank];
			int* recvDisplsT = new int[nRank];
			MPI_Datatype* recvTypes = new MPI_Datatype[nRank];

			int* typesVals = new int[nDim];
			int* blockSizeVals = new int[nDim];
			int* typeValsMaxIndex = new int[nDim];

			int* blocks = GetBlocks();
			for (int i = 0; i < nDim; i++)
			{
				blockSizeVals[i] = dimsGlobal[i] / blocks[i];
				if (dimsGlobal[i] % blocks[i] == 0)
				{
					typesVals[i] = 1;
				}
				else
				{
					typesVals[i] = 2;
				}

				typeValsMaxIndex[i] = dimsGlobal[i] % blocks[i];
			}

			int typesSize = 1;
			for (int i = 0; i < nDim; i++)
			{
				typesSize = typesSize * typesVals[i];
			}

			MPI_Datatype* blockTypes = new MPI_Datatype[typesSize];

			int* dimsGlobalReverse = new int[nDim];
			for (int i = 0; i < nDim; i++)
			{
				dimsGlobalReverse[i] = dimsGlobal[nDim - 1 - i];
			}

			int* subSizes = new int[nDim];
			int* starts = new int[nDim];
			for (int i = 0; i < nDim; i++)
			{
				starts[i] = 0;
			}

			if (nDim == DIM_1D)
			{
				for (int i = 0; i < typesVals[X]; i++)
				{
					subSizes[X] = blockSizeVals[X] + i;
					MPI_Type_create_subarray(nDim, dimsGlobalReverse, subSizes, starts, MPI_ORDER_C, dataType, &blockTypes[i]);
					MPI_Type_commit(&blockTypes[i]);
				}
			}
			else if (nDim == DIM_2D)
			{
				for (int j = 0; j < typesVals[Y]; j++)
				{
					subSizes[X] = blockSizeVals[Y] + j;
					for (int i = 0; i < typesVals[X]; i++)
					{
						subSizes[Y] = blockSizeVals[X] + i;

						MPI_Type_create_subarray(nDim, dimsGlobalReverse, subSizes, starts, MPI_ORDER_C, dataType, &blockTypes[i + typesVals[X] * j]);
						MPI_Type_commit(&blockTypes[i + typesVals[X] * j]);
					}
				}
			}
			else if (nDim == DIM_3D)
			{
				for (int k = 0; k < typesVals[Z]; k++)
				{
					subSizes[X] = blockSizeVals[Z] + k;
					for (int j = 0; j < typesVals[Y]; j++)
					{
						subSizes[Y] = blockSizeVals[Y] + j;
						for (int i = 0; i < typesVals[X]; i++)
						{
							subSizes[Z] = blockSizeVals[X] + i;

							MPI_Type_create_subarray(nDim, dimsGlobalReverse, subSizes, starts, MPI_ORDER_C, dataType, &blockTypes[i + typesVals[X] * j + typesVals[X] * typesVals[Y] * k]);
							MPI_Type_commit(&blockTypes[i + typesVals[X] * j + typesVals[X] * typesVals[Y] * k]);
						}
					}
				}
			}

			for (int proc = 0; proc < nRank; proc++)
			{
				int* coords = new int[nDim];
				int* dimValsLocal = new int[nDim];
				int* dimValsLocalT = new int[nDim];

				MPI_Cart_coords(comm, proc, nDim, coords);
				GetLocalPartitionDims(nDim, coords, dimsGlobal, dimValsLocalT);

				for (int i = 0; i < nDim; i++)
				{
					dimValsLocal[i] = dimValsLocalT[i] - 2 * ghostLayer;
				}

				int displsIndex = 0;

				int displsXIndex = 0;
				if (typeValsMaxIndex[X] == 0 || coords[X] < typeValsMaxIndex[X])
				{
					displsXIndex += (coords[X] * dimValsLocal[X]);
				}
				else
				{
					displsXIndex += ((typeValsMaxIndex[X]) * (dimValsLocal[X] + 1) + (coords[X] - typeValsMaxIndex[X]) * (dimValsLocal[X]));
				}

				displsIndex = displsXIndex;

				if (nDim == DIM_2D || nDim == DIM_3D)
				{
					int displsYIndex = 0;
					if (typeValsMaxIndex[Y] == 0 || coords[Y] < typeValsMaxIndex[Y])
					{
						displsYIndex += (coords[Y] * dimValsLocal[Y]);
					}
					else
					{
						displsYIndex += ((typeValsMaxIndex[Y]) * (dimValsLocal[Y] + 1) + (coords[Y] - typeValsMaxIndex[Y]) * (dimValsLocal[Y]));
					}

					displsIndex += dimsGlobal[X] * displsYIndex;
				}

				if (nDim == DIM_3D)
				{
					int displsZIndex = 0;
					if (typeValsMaxIndex[Z] == 0 || coords[Z] < typeValsMaxIndex[Z])
					{
						displsZIndex += (coords[Z] * dimValsLocal[Z]);
					}
					else
					{
						displsZIndex += ((typeValsMaxIndex[Z]) * (dimValsLocal[Z] + 1) + (coords[Z] - typeValsMaxIndex[Z]) * (dimValsLocal[Z]));
					}

					displsIndex += dimsGlobal[X] * dimsGlobal[Y] * displsZIndex;
				}

				recvCounts[proc] = 1;
				recvDisplsT[proc] = displsIndex;
				recvTypes[proc] = blockTypes[GetBlockTypeIndex(nDim, dimValsLocal, typesVals, blockSizeVals)];

				delete[] coords;
				delete[] dimValsLocal;
				delete[] dimValsLocalT;
			}

			MPI_Status* status = new MPI_Status[nRank];
			MPI_Request* request = new MPI_Request[nRank];
			T* globalData = global->GetData();

			for (int proc = 0; proc < nRank; proc++)
			{
				if (proc == GetMasterRank())
				{
					T* localData = local->GetData();
					MPI_Sendrecv(&(localData[sendDispls]), sendCounts, sendTypes, proc, 0,
						&(globalData[recvDisplsT[proc]]), recvCounts[proc], recvTypes[proc], proc, 0,
						comm, &status[proc]);
				}
				else
				{
					MPI_Irecv(&(globalData[recvDisplsT[proc]]), recvCounts[proc], recvTypes[proc], proc, 0, comm, &request[proc]);
				}
			}

			for (int proc = 1; proc < nRank; proc++)
			{
				MPI_Wait(&request[proc], &status[proc]);
			}

			for (int i = 0; i < typesSize; i++)
			{
				MPI_Type_free(&blockTypes[i]);
			}

			delete[] recvCounts;
			delete[] recvDisplsT;
			delete[] recvTypes;

			delete[] typesVals;
			delete[] blockSizeVals;
			delete[] typeValsMaxIndex;

			delete[] blockTypes;
			delete[] dimsGlobalReverse;

			delete[] subSizes;
			delete[] starts;

			delete[] status;
			delete[] request;
		}
		else
		{
			MPI_Status sendStatus;
			MPI_Request sendRequest;
			T* localData = local->GetData();
			MPI_Isend(&(localData[sendDispls]), sendCounts, sendTypes, GetMasterRank(), 0, comm, &sendRequest);
			MPI_Wait(&sendRequest, &sendStatus);
		}

		MPI_Type_free(&sendTypes);

		MPI_Barrier(comm);

		delete[] dimsLocalReverse;
		delete[] dimsLocalReverse2;
		delete[] startsS;
	}


	template<typename T>
	void Exchange(Array<T>* arr, HaloBuffer<T>* buffer)
	{
		MPI_Datatype dataType = MPI_BYTE;
		int tTypeSize = sizeof(T);
		MPI_Status status;
		MPI_Comm& comm = GetCommunicator();

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
					CopyFromArrayToBuffer(arr, buffer, face1);
				}
				MPI_Sendrecv(&(buffer->GetSendBuffer(face1)[0]), buffer->GetHaloSize(face1) * tTypeSize, dataType, neighbour1, 0,
					&(buffer->GetRecvBuffer(face2)[0]), buffer->GetHaloSize(face2) * tTypeSize, dataType, neighbour2, 0,
					comm, &status);
				if (neighbour2 != MPI_PROC_NULL)
				{
					CopyFromBufferToArray(arr, buffer, face2);
				}

				if (neighbour2 != MPI_PROC_NULL)
				{
					CopyFromArrayToBuffer(arr, buffer, face2);
				}
				MPI_Sendrecv(&(buffer->GetSendBuffer(face2)[0]), buffer->GetHaloSize(face2) * tTypeSize, dataType, neighbour2, 0,
					&(buffer->GetRecvBuffer(face1)[0]), buffer->GetHaloSize(face1) * tTypeSize, dataType, neighbour1, 0,
					comm, &status);
				if (neighbour1 != MPI_PROC_NULL)
				{
					CopyFromBufferToArray(arr, buffer, face1);
				}
			}
		}
	}


#ifdef APPFIS_CUDA
	template<typename T>
	void ExchangeGPU(Array<T>* arr, HaloBuffer<T>* buffer)
	{
		MPI_Datatype dataType = MPI_BYTE;
		int tTypeSize = sizeof(T);
		MPI_Status status;
		MPI_Comm& comm = GetCommunicator();

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

				if (neighbour1 != MPI_PROC_NULL && i > 0)
				{
					CopyFromArrayToBufferBorder(arr, buffer, face1);
				}
				MPI_Sendrecv(&(buffer->GetSendBuffer(face1)[0]), buffer->GetHaloSize(face1) * tTypeSize, dataType, neighbour1, 0,
					&(buffer->GetRecvBuffer(face2)[0]), buffer->GetHaloSize(face2) * tTypeSize, dataType, neighbour2, 0,
					comm, &status);
				if (neighbour2 != MPI_PROC_NULL)
				{
					CopyFromBufferToArrayBorder(arr, buffer, face2);
				}

				if (neighbour2 != MPI_PROC_NULL && i > 0)
				{
					CopyFromArrayToBufferBorder(arr, buffer, face2);
				}
				MPI_Sendrecv(&(buffer->GetSendBuffer(face2)[0]), buffer->GetHaloSize(face2) * tTypeSize, dataType, neighbour2, 0,
					&(buffer->GetRecvBuffer(face1)[0]), buffer->GetHaloSize(face1) * tTypeSize, dataType, neighbour1, 0,
					comm, &status);
				if (neighbour1 != MPI_PROC_NULL)
				{
					CopyFromBufferToArrayBorder(arr, buffer, face1);
				}
			}
		}
	}
#endif


	template<typename T>
	T AllReduce(T local, REDUCTION rType)
	{
		MPI_Datatype dataType;
		if (std::is_same<T, int>::value)
		{
			dataType = MPI_INT;
		}
		else if (std::is_same<T, double>::value)
		{
			dataType = MPI_DOUBLE;
		}
		else if (std::is_same<T, float>::value)
		{
			dataType = MPI_FLOAT;
		}
		else
		{
			std::cerr << "Partition failed. Not supported MPI datatype" << std::endl;
			exit(EXIT_FAILURE);
		}
		MPI_Comm& comm = GetCommunicator();

		T global = 0;
		if (rType == MAX)
		{
			MPI_Allreduce(&local, &global, 1, dataType, MPI_MAX, comm);
		}
		else if (rType == MIN)
		{
			MPI_Allreduce(&local, &global, 1, dataType, MPI_MIN, comm);
		}
		else if (rType == SUM)
		{
			MPI_Allreduce(&local, &global, 1, dataType, MPI_SUM, comm);
		}
		return global;
	}


	template<typename T>
	void ExchangeInitial(Array<T>* arr, HaloBuffer<T>* buffer)
	{
		Exchange(arr, buffer);
		MPI_Barrier(GetCommunicator());
	}


	int GetBlockTypeIndex(int nDim, int* dimValsLocal, int* typesVals, int* blockSizeVals)
	{
		int* diff = new int[nDim];
		for (int i = 0; i < nDim; i++)
		{
			if (dimValsLocal[i] != blockSizeVals[i])
			{
				diff[i] = 1;
			}
			else
			{
				diff[i] = 0;
			}
		}

		int val = 0;
		if (nDim == DIM_1D) val = diff[X];
		else if (nDim == DIM_2D) val = diff[X] + typesVals[X] * diff[Y];
		else if (nDim == DIM_3D) val = diff[X] + typesVals[X] * diff[Y] + typesVals[X] * typesVals[Y] * diff[Z];

		delete[] diff;
		return val;
	}


	void ProcessSync()
	{
		MPI_Barrier(GetCommunicator());
	}

}


#endif