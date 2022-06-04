3D Heat Diffusion Solver using Forward Time Centered Space (FTCS).

Compile:
	make //Serial version
	make config=cuda //CUDA version 
	make config=acpu //CPU version using APPFIS (MPI and OpenMP)
	make config=acuda //CUDA version using APPFIS (MPI and CUDA)

Run:
	./build/heat3d -dimx=256 -dimy=256 -dimz=256 -iteration=1000
	mpirun -n 1 ./build/heat3d -dimx=256 -dimy=256 -dimz=256 -iteration=1000

Optional: -thread=1 //OpenMP thread per process
		  -overlap	//Computation communication overlapping
		  -output	//Print max and min heat