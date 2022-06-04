/**
 *
 * 3D Heat Diffusion Solver using Forward Time Centered Space (FTCS).
 * A Gaussian Distribution for Initial Data
 * Boundary Conditions include Dirichlet (constant boundary value = 0)
 * Author: Md Bulbul Sharif
 *
 **/


#if defined(APPFIS_CUDA) || defined(APPFIS_CPU)
#include "heat_appfis.hpp"
#elif defined(CUDA)
#include "heat_cuda.hpp"
#else
#include "heat_serial.hpp"
#endif


int main(int argc, char* argv[])
{
	HeatDiffusion(argc, argv);

	return 0;
}
