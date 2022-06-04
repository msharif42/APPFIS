A finite differences time domain progression stencil on a 3D surface.

Adapted from Nvidia samples (https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/FDTD3d)

Compile:
	make //Serial version
	make config=cuda //CUDA version 
	make config=acpu //CPU version using APPFIS (MPI and OpenMP)
	make config=acuda //CUDA version using APPFIS (MPI and CUDA)

Run:
	./build/fdtd3d -dimx=256 -dimy=256 -dimz=256 -iteration=1000
	mpirun -n 1 ./build/fdtd3d -dimx=256 -dimy=256 -dimz=256 -iteration=1000

Optional: -thread=1 //OpenMP thread per process
		  -overlap	//Computation communication overlapping
		  -output	//Save the output in a ascii file