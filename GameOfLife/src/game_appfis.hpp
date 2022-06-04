/**
 *
 * APPFIS version of Game Of Life
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef GAME_APPFIS_HPP
#define GAME_APPFIS_HPP


#include <iostream>
#include "../../Appfis/Appfis.hpp"
using namespace std;


namespace APPFIS
{
	PARALLEL void STENCIL(const int i, const int j, APPFIS::Array<int>* cur, APPFIS::Array<int>* next);
}
void GameOfLife(int argc, char* argv[]);
void InitialData(int dimX, int dimY, APPFIS::Array<int>* data);


void GameOfLife(int argc, char* argv[])
{
	APPFIS::Timer timer = APPFIS::Timer();
	timer.Start("TOTAL");

	int dimX = 64;
	int dimY = 64;
	int iteration = 10;

	int thread = 1;
	bool overlap = false;
	bool outputFlag = false;

	if (APPFIS::CheckArgument(argc, argv, "dimx")) {
		dimX = APPFIS::GetArgument(argc, argv, "dimx");
	}
	if (APPFIS::CheckArgument(argc, argv, "dimy")) {
		dimY = APPFIS::GetArgument(argc, argv, "dimy");
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
	attribute.periodic = true;

	APPFIS::Initialize(argc, argv, APPFIS::DIM_2D, APPFIS::DIM_2D, attribute);

	APPFIS::Grid<int> g(APPFIS::DIM_2D, dimX, dimY, APPFIS::DUAL);
	if (APPFIS::IsMaster())
	{
		g.AllocateFullGrid();
		InitialData(dimX, dimY, g.GetFullGrid());
	}

	APPFIS::Scatter(g.GetFullGrid(), g.GetSubGrid());
	APPFIS::Config<int> config(g);
	config.GridsToUpdate(g);
	APPFIS::InitializeExecution(g);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		APPFIS::Execute2D(config, g.GetSubGrid(), g.GetSubGridNext());
	}
	timer.Stop("COMPUTE");

	if (outputFlag)
	{
		int liveCount = 0;
		APPFIS::Execute2DReduce(config, liveCount, APPFIS::SUM, g);
		if (APPFIS::IsMaster())
		{
			std::cout << "Live Count: " << liveCount << std::endl;
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
		std::cout << "GFLOPS: " << 7 * timer.CalculateBLUPS("COMPUTE", dimX * dimY, iteration) << std::endl;
	}

	APPFIS::Finalize();
}


void InitialData(int dimX, int dimY, APPFIS::Array<int>* data)
{
	srand(0);
	for (int j = 0; j < dimY; j++)
	{
		for (int i = 0; i < dimX; i++)
		{
			data->Set(i, j, rand() % 2);
		}
	}
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, APPFIS::Array<int>* cur, APPFIS::Array<int>* next)
{
	int count = cur->Get(i - 1, j - 1) + cur->Get(i, j - 1) + cur->Get(i + 1, j - 1)
		+ cur->Get(i - 1, j) + cur->Get(i + 1, j)
		+ cur->Get(i - 1, j + 1) + cur->Get(i, j + 1) + cur->Get(i + 1, j + 1);

	if (cur->Get(i, j) == 1)
	{
		if (count < 2 || count > 4) next->Set(i, j, 0);
	}
	else
	{
		if (count == 3) next->Set(i, j, 1);
	}
}


#endif