/**
 *
 * APPFIS version of FDTD3d
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef FDTD3D_APPFIS_HPP
#define FDTD3D_APPFIS_HPP


#include <iostream>
#include "../../Appfis/Appfis.hpp"
using namespace std;

namespace APPFIS
{
	PARALLEL void STENCIL(const int i, const int j, const int k, float coeff, int radius, APPFIS::Array<float>* cur, APPFIS::Array<float>* next);
}
void FDTD3d(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int dimZ, APPFIS::Array<float>* data);


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

	int thread = 1;
	bool overlap = false;
	bool outputFlag = false;

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
	if (APPFIS::CheckArgument(argc, argv, "thread")) {
		thread = APPFIS::GetArgument(argc, argv, "thread");
	}
	overlap = APPFIS::CheckArgument(argc, argv, "overlap");
	outputFlag = APPFIS::CheckArgument(argc, argv, "output");

	APPFIS::Attribute attribute;
	attribute.threads = thread;
	attribute.overlap = overlap;
	attribute.ghostLayer = radius;
	attribute.periodic = false;

	APPFIS::Initialize(argc, argv, APPFIS::DIM_3D, APPFIS::DIM_3D, attribute);

	APPFIS::Grid<float> g(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::DUAL);
	if (APPFIS::IsMaster())
	{
		g.AllocateFullGrid();
		InitialData(dimX, dimY, dimZ, g.GetFullGrid());
	}

	APPFIS::Scatter(g.GetFullGrid(), g.GetSubGrid());
	APPFIS::Config<float> config(g);
	config.GridsToUpdate(g);
	APPFIS::InitializeExecution(g);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		APPFIS::Execute3D(config, coeff, radius, g.GetSubGrid(), g.GetSubGridNext());
	}
	timer.Stop("COMPUTE");

	APPFIS::FinalizeExecution(g);
	APPFIS::Gather(g.GetFullGrid(), g.GetSubGrid());

	timer.Stop("TOTAL");
	if (APPFIS::IsMaster())
	{
		double totalTime = timer.GetCustomTime("TOTAL");
		double computeTime = timer.GetCustomTime("COMPUTE");
		std::cout << "Total: " << totalTime << std::endl;
		std::cout << "Compute: " << computeTime << std::endl;
		std::cout << "GFLOPS: " << 25 * timer.CalculateBLUPS("COMPUTE", dimX * dimY * dimZ, iteration) << std::endl;

		if (outputFlag)
		{
			std::string fileName("./output/output.txt");
			g.GetFullGrid()->SaveAsciiFile(fileName);
		}
	}

	APPFIS::Finalize();
}


void InitialData(int dimX, int dimY, int dimZ, APPFIS::Array<float>* data)
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
				data->Set(x, y, z, val);
			}
		}
	}
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, const int k, float coeff, int radius, APPFIS::Array<float>* cur, APPFIS::Array<float>* next)
{
	float val = cur->Get(i, j, k);

	for (int a = 1; a <= radius; a++)
	{
		val += (cur->Get(i - a, j, k) + cur->Get(i + a, j, k))
			+ (cur->Get(i, j - a, k) + cur->Get(i, j + a, k))
			+ (cur->Get(i, j, k - a) + cur->Get(i, j, k + a));
	}

	val *= coeff;

	next->Set(i, j, k, val);
}


#endif