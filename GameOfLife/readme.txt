2D Game of Life

Compile:
	make //Serial version
	make config=cuda //CUDA version 
	make config=acpu //CPU version using APPFIS (MPI and OpenMP)
	make config=acuda //CUDA version using APPFIS (MPI and CUDA)

Run:
	./build/game2d -dimx=2048 -dimy=2048 -iteration=1000
	mpirun -n 1 ./build/game2d -dimx=2048 -dimy=2048 -iteration=1000

Optional: -thread=1 //OpenMP thread per process
		  -overlap	//Computation communication overlapping
		  -output	//Print live cell count