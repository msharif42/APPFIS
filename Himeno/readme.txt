Himeno benchmark

Compile:
	make config=acpu //CPU version using APPFIS (MPI and OpenMP)
	make config=acuda //CUDA version using APPFIS (MPI and CUDA)

Run:
	mpirun -n 1 ./build/himeno -dimx=256 -dimy=256 -dimz=256 -iteration=1000

Optional: -thread=1 //OpenMP thread per process
		  -overlap	//Computation communication overlapping
		  -output	//Print sum