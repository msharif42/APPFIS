/**
 *
 * APPFIS version of Heat Diffusion
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef HEAT_APPFIS_HPP
#define HEAT_APPFIS_HPP


#include <iostream>
#include "../../Appfis/Appfis.hpp"
using namespace std;

namespace APPFIS
{
	PARALLEL void STENCIL(const int i, const int j, const int k, double cx, double cy, double cz, APPFIS::Array<double>* cur, APPFIS::Array<double>* next);
}
void HeatDiffusion(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int dimZ, APPFIS::Array<double>* data);


void HeatDiffusion(int argc, char* argv[])
{
	APPFIS::Timer timer = APPFIS::Timer();
	timer.Start("TOTAL");

	int dimX = 64;
	int dimY = 64;
	int dimZ = 64;
	int iteration = 10;
	double dt = 0.1;
	double dc = 0.01;
	double dx, dy, dz;
	dx = dy = dz = 0.1;
	double cx = (dt * dc) / pow(dx, 2);
	double cy = (dt * dc) / pow(dy, 2);
	double cz = (dt * dc) / pow(dz, 2);

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
	attribute.ghostLayer = 1;
	attribute.periodic = false;

	APPFIS::Initialize(argc, argv, APPFIS::DIM_3D, APPFIS::DIM_3D, attribute);

	APPFIS::Grid<double> g(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::DUAL);

	if (APPFIS::IsMaster())
	{
		g.AllocateFullGrid();
		InitialData(dimX, dimY, dimZ, g.GetFullGrid());
	}

	APPFIS::Scatter(g.GetFullGrid(), g.GetSubGrid());
	g.SetBoundary(0.0);
	APPFIS::Config<double> config(g);
	config.GridsToUpdate(g);
	APPFIS::InitializeExecution(g);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		APPFIS::Execute3D(config, cx, cy, cz, g.GetSubGrid(), g.GetSubGridNext());
	}
	timer.Stop("COMPUTE");

	if (outputFlag)
	{
		double maxHeat = 0;
		double minHeat = 0;
		APPFIS::Execute3DReduce(config, maxHeat, APPFIS::MAX, g);
		APPFIS::Execute3DReduce(config, minHeat, APPFIS::MIN, g);
		if (APPFIS::IsMaster())
		{
			std::cout << "Max: " << maxHeat << std::endl;
			std::cout << "Min: " << minHeat << std::endl;
		}
	}

	APPFIS::FinalizeExecution(g);
	APPFIS::Gather(g.GetFullGrid(), g.GetSubGrid());

	timer.Stop("TOTAL");
	if (APPFIS::IsMaster())
	{
		double totalTime = timer.GetCustomTime("TOTAL");
		double computeTime = timer.GetCustomTime("COMPUTE");
		std::cout << "Total: " << totalTime << std::endl;
		std::cout << "Compute: " << computeTime << std::endl;
		std::cout << "GFLOPS: " << 12 * timer.CalculateBLUPS("COMPUTE", dimX * dimY * dimZ, iteration) << std::endl;
	}

	APPFIS::Finalize();
}


void InitialData(int dimX, int dimY, int dimZ, APPFIS::Array<double>* data)
{
	double meanX = (dimX - 1) / 2.0;
	double meanY = (dimY - 1) / 2.0;
	double meanZ = (dimZ - 1) / 2.0;

	double sumX = 0.0;
	double sumY = 0.0;
	double sumZ = 0.0;

	for (int i = 0; i < dimX; i++)
	{
		sumX += pow(i - meanX, 2);
	}

	for (int j = 0; j < dimY; j++)
	{
		sumY += pow(j - meanY, 2);
	}

	for (int k = 0; k < dimZ; k++)
	{
		sumZ += pow(k - meanZ, 2);
	}

	double varX = sumX / (dimX - 1);
	double varY = sumY / (dimY - 1);
	double varZ = sumZ / (dimZ - 1);

	for (int k = 0; k < dimZ; k++)
	{
		for (int j = 0; j < dimY; j++)
		{
			for (int i = 0; i < dimX; i++)
			{
				double val = exp(-(pow(i - meanX, 2) / (2 * varX) + pow(j - meanY, 2) / (2 * varY) + pow(k - meanZ, 2) / (2 * varZ)));
				data->Set(i, j, k, val);
			}
		}
	}
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, const int k, double cx, double cy, double cz, APPFIS::Array<double>* cur, APPFIS::Array<double>* next)
{
	double val = cur->Get(i, j, k)
		+ cx * (cur->Get(i - 1, j, k) + cur->Get(i + 1, j, k) - 2 * cur->Get(i, j, k))
		+ cy * (cur->Get(i, j - 1, k) + cur->Get(i, j + 1, k) - 2 * cur->Get(i, j, k))
		+ cz * (cur->Get(i, j, k - 1) + cur->Get(i, j, k + 1) - 2 * cur->Get(i, j, k));

	next->Set(i, j, k, val);
}


#endif