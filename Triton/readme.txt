Triton

Compile:
	make config=acpu //CPU version using APPFIS (MPI and OpenMP)
	make config=acuda //CUDA version using APPFIS (MPI and CUDA)

Run:
	mpirun -n 1 ./build/triton ./input/cfg/case01.cfg

Optional: -thread=1 //OpenMP thread per process
		  -overlap	//Computation communication overlapping
		  -output   //To save h,u and v in file