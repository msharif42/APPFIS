/**
 *
 * Triton
 * Author: Md Bulbul Sharif
 *
 **/


#if defined(APPFIS_CUDA) || defined(APPFIS_CPU)
#include "triton_appfis.hpp"
#endif


int main(int argc, char* argv[])
{
#if defined(APPFIS_CUDA) || defined(APPFIS_CPU)
	Triton(argc, argv);
#endif

	return 0;
}
