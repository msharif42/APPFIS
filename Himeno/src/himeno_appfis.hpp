/**
 *
 * APPFIS version of Himeno
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef HIMENO_APPFIS_HPP
#define HIMENO_APPFIS_HPP


#include <iostream>
#include "../../Appfis/Appfis.hpp"
using namespace std;

namespace APPFIS
{
	PARALLEL void STENCIL(const int i, const int j, const int k, float omega, 
		APPFIS::Array<float>* a0, APPFIS::Array<float>* a1, APPFIS::Array<float>* a2, APPFIS::Array<float>* a3,
		APPFIS::Array<float>* b0, APPFIS::Array<float>* b1, APPFIS::Array<float>* b2,
		APPFIS::Array<float>* c0, APPFIS::Array<float>* c1, APPFIS::Array<float>* c2,
		APPFIS::Array<float>* bnd, APPFIS::Array<float>* wrk1, APPFIS::Array<float>* p, APPFIS::Array<float>* wrk2);
}
void Himeno(int argc, char* argv[]);
void InitialData(int dimX, int dimY, int dimZ, APPFIS::Array<float>* data);
void InitialSubs(APPFIS::Array<float>* a0, APPFIS::Array<float>* a1, APPFIS::Array<float>* a2, APPFIS::Array<float>* a3,
	APPFIS::Array<float>* b0, APPFIS::Array<float>* b1, APPFIS::Array<float>* b2,
	APPFIS::Array<float>* c0, APPFIS::Array<float>* c1, APPFIS::Array<float>* c2,
	APPFIS::Array<float>* bnd, APPFIS::Array<float>* wrk1);


void Himeno(int argc, char* argv[])
{
	APPFIS::Timer timer = APPFIS::Timer();
	timer.Start("TOTAL");

	int dimX = 64;
	int dimY = 64;
	int dimZ = 64;
	int iteration = 10;
	float omega = 0.8f;

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

	APPFIS::Grid<float> gp(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::DUAL);
	APPFIS::Grid<float> ga0(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> ga1(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> ga2(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> ga3(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gb0(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gb1(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gb2(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gc0(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gc1(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gc2(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gbnd(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);
	APPFIS::Grid<float> gwrk1(APPFIS::DIM_3D, dimX, dimY, dimZ, APPFIS::SINGLE);

	if (APPFIS::IsMaster())
	{
		gp.AllocateFullGrid();
		InitialData(dimX, dimY, dimZ, gp.GetFullGrid());
	}

	APPFIS::Scatter(gp.GetFullGrid(), gp.GetSubGrid());
	InitialSubs(ga0.GetSubGrid(), ga1.GetSubGrid(), ga2.GetSubGrid(), ga3.GetSubGrid(),
		gb0.GetSubGrid(), gb1.GetSubGrid(), gb2.GetSubGrid(),
		gc0.GetSubGrid(), gc1.GetSubGrid(), gc2.GetSubGrid(),
		gbnd.GetSubGrid(), gwrk1.GetSubGrid());

	APPFIS::Config<float> config(gp);
	config.GridsToUpdate(gp);
	APPFIS::InitializeExecution(gp, ga0, ga1, ga2, ga3, gb0, gb1, gb2, gc0, gc1, gc2, gbnd, gwrk1);

	timer.Start("COMPUTE");
	for (int i = 0; i < iteration; i++)
	{
		APPFIS::Execute3D(config, omega,
			ga0.GetSubGrid(), ga1.GetSubGrid(), ga2.GetSubGrid(), ga3.GetSubGrid(),
			gb0.GetSubGrid(), gb1.GetSubGrid(), gb2.GetSubGrid(),
			gc0.GetSubGrid(), gc1.GetSubGrid(), gc2.GetSubGrid(),
			gbnd.GetSubGrid(), gwrk1.GetSubGrid(), gp.GetSubGrid(), gp.GetSubGridNext());
	}
	timer.Stop("COMPUTE");

	if (outputFlag)
	{
		float sum = 0;
		APPFIS::Execute3DReduce(config, sum, APPFIS::SUM, gp);
		if (APPFIS::IsMaster())
		{
			std::cout << "Sum: " << sum << std::endl;
		}
	}

	APPFIS::FinalizeExecution(gp);
	APPFIS::Gather(gp.GetFullGrid(), gp.GetSubGrid());

	timer.Stop("TOTAL");
	if (APPFIS::IsMaster())
	{
		double totalTime = timer.GetCustomTime("TOTAL");
		double computeTime = timer.GetCustomTime("COMPUTE");
		std::cout << "Total: " << totalTime << std::endl;
		std::cout << "Compute: " << computeTime << std::endl;
		std::cout << "GFLOPS: " << 32 * timer.CalculateBLUPS("COMPUTE", dimX * dimY * dimZ, iteration) << std::endl;
	}

	APPFIS::Finalize();
}


void InitialData(int dimX, int dimY, int dimZ, APPFIS::Array<float>* data)
{
	for (int k = 0; k < dimZ; k++)
	{
		for (int j = 0; j < dimY; j++)
		{
			for (int i = 0; i < dimX; i++)
			{
				float val = (float)((i + 1) * (i+1)) / (float)(dimX * dimX);
				data->Set(i, j, k, val);
			}
		}
	}
}


void InitialSubs(APPFIS::Array<float>* a0, APPFIS::Array<float>* a1, APPFIS::Array<float>* a2, APPFIS::Array<float>* a3,
	APPFIS::Array<float>* b0, APPFIS::Array<float>* b1, APPFIS::Array<float>* b2,
	APPFIS::Array<float>* c0, APPFIS::Array<float>* c1, APPFIS::Array<float>* c2,
	APPFIS::Array<float>* bnd, APPFIS::Array<float>* wrk1)
{
	a0->FillData(1.0);
	a1->FillData(1.0);
	a2->FillData(1.0);
	a3->FillData(1.0f/6.0f);
	b0->FillData(0.0);
	b1->FillData(0.0);
	b2->FillData(0.0);
	c0->FillData(1.0);
	c1->FillData(1.0);
	c2->FillData(1.0);
	bnd->FillData(1.0);
	wrk1->FillData(0.0);
}


PARALLEL void APPFIS::STENCIL(const int i, const int j, const int k, float omega,
	APPFIS::Array<float>* a0, APPFIS::Array<float>* a1, APPFIS::Array<float>* a2, APPFIS::Array<float>* a3,
	APPFIS::Array<float>* b0, APPFIS::Array<float>* b1, APPFIS::Array<float>* b2,
	APPFIS::Array<float>* c0, APPFIS::Array<float>* c1, APPFIS::Array<float>* c2,
	APPFIS::Array<float>* bnd, APPFIS::Array<float>* wrk1, APPFIS::Array<float>* p, APPFIS::Array<float>* wrk2) 
{
	float s0 = a0->Get(i, j, k) * p->Get(i + 1, j, k)
		+ a1->Get(i, j, k) * p->Get(i, j + 1, k)
		+ a2->Get(i, j, k) * p->Get(i, j, k + 1)
		+ b0->Get(i, j, k) * (p->Get(i + 1, j + 1, k) - p->Get(i + 1, j - 1, k) - p->Get(i - 1, j + 1, k) + p->Get(i - 1, j - 1, k))
		+ b1->Get(i, j, k) * (p->Get(i, j + 1, k + 1) - p->Get(i, j - 1, k + 1) - p->Get(i, j + 1, k - 1) + p->Get(i, j - 1, k - 1))
		+ b2->Get(i, j, k) * (p->Get(i + 1, j, k + 1) - p->Get(i - 1, j, k + 1) - p->Get(i + 1, j, k - 1) + p->Get(i - 1, j, k - 1))
		+ c0->Get(i, j, k) * p->Get(i - 1, j, k)
		+ c1->Get(i, j, k) * p->Get(i, j - 1, k)
		+ c2->Get(i, j, k) * p->Get(i, j, k - 1)
		+ wrk1->Get(i, j, k);

	float ss = (s0 * a3->Get(i, j, k) - p->Get(i, j, k)) * bnd->Get(i, j, k);
	float val = p->Get(i, j, k) + omega * ss;
	wrk2->Set(i, j, k, val);
}


#endif