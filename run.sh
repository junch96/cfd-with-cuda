g++ -std=c++11 -c glUtil.cpp -Iglad/include
g++ -std=c++11 -c main.cpp -Iglad/include
g++ -std=c++11 -c glad.cpp -Iglad/include
nvcc -c solver.cu -Iglad/include
nvcc main.o glUtil.o glad.o solver.o -o main -lglfw
./main
rm *.o