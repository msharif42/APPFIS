/**
 *
 * Main file from where Game Of Life execution starts
 * Author: Md Bulbul Sharif
 *
 **/


#if defined(APPFIS_CUDA) || defined(APPFIS_CPU)
#include "game_appfis.hpp"
#elif defined(CUDA)
#include "game_cuda.hpp"
#else
#include "game_serial.hpp"
#endif


int main(int argc, char* argv[])
{
	GameOfLife(argc, argv);

	return 0;
}