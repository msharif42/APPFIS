/**
 *
 * Himeno benchmark
 * Author: Md Bulbul Sharif
 *
 **/


#if defined(APPFIS_CUDA) || defined(APPFIS_CPU)
#include "himeno_appfis.hpp"
#endif


int main(int argc, char* argv[])
{
#if defined(APPFIS_CUDA) || defined(APPFIS_CPU)
	Himeno(argc, argv);
#endif

	return 0;
}
