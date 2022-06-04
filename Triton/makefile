ifeq ($(config), acuda)
	CC := nvcc
	FLAGS := --compiler-bindir `which mpicxx` -x cu -Xcompiler -fopenmp 
	DFLAGS := -DAPPFIS_CUDA=1
else ifeq ($(config), acpu)
	CC := mpicxx 
	FLAGS := -Wall -fopenmp 
	DFLAGS := -DAPPFIS_CPU=1
else ifeq ($(config), cuda)
	CC := nvcc
	FLAGS := -x cu 
	DFLAGS := -DCUDA=1
else
	CC := g++ 
	FLAGS := 
	DFLAGS := -DCPU=1
endif


triton: ./src/main.cpp
	if [ ! -d "build" ]; then mkdir build; fi
	@echo 'Compiling file: $<'
	$(CC) $(INC_DIRS:%=-I%) $(FLAGS) $(DFLAGS) -O3 $(LIBRARIES) -o "build/$@" "$<" -std=c++11
	@echo 'Building finished: $@'
	
clean:
	rm -rf ./build/triton