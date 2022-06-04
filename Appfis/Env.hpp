/**
 *
 * This file contains all the environment variables used in APPFIS
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef ENV_HPP
#define ENV_HPP


namespace APPFIS
{

#ifdef APPFIS_CUDA
#define PARALLEL __device__
#define EXEC_SPACE __host__ __device__
#else
#define PARALLEL inline
#define EXEC_SPACE inline
#endif

#define STENCIL Stencil

	bool EXECUTION_FLAG = false;

	enum AXIS { X, Y, Z };
	enum DIMENSION { DIM_1D = 1, DIM_2D = 2, DIM_3D = 3 };
	enum DIRECTION { LEFT, RIGHT, UP, DOWN, FRONT, BACK };
	enum GRID_TYPE { SINGLE, DUAL, FLAT };
	enum REDUCTION { MAX, MIN, SUM };

}


#endif