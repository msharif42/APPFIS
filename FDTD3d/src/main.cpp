/**
 *
 * This example applies a finite differences time domain progression stencil on a 3D surface.
 * We have adapted this from Nvidia samples (https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/FDTD3d)
 * Author: Md Bulbul Sharif
 *
 **/


#if defined(APPFIS_CUDA) || defined(APPFIS_CPU)
#include "fdtd3d_appfis.hpp"
#elif defined(CUDA)
#include "fdtd3d_cuda.hpp"
#else
#include "fdtd3d_serial.hpp"
#endif


int main(int argc, char* argv[])
{
	FDTD3d(argc, argv);

	return 0;
}