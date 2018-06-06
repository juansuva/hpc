# HPC TSP-CUDA
Travelling salesman problem in CUDA using permutations to obtain better sub-optimal solutions.
Sample files for input are in cities directory


Compile mainCPU.cpp:
```
g++ mainCPU.cpp -o mainCPU -fopenmp -O2
```
Compile mainCUDA.cu:
```
nvcc mainCUDA.cc -o mainCUDA -Xcompiler -fopenmp -O2
```

[paper is here:](https://es.sharelatex.com/read/ngpqwphrcfpj)


