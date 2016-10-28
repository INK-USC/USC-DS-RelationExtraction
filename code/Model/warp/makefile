CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
LIBS = -L/usr/lib/x86_64-linux-gnu

warp warp-debug: warp.o warp-debug.o
	$(CC) $(CFLAGS) -o warp warp.o $(LIBS) -lgsl -lm -lgslcblas
	$(CC) $(CFLAGS) -o warp-debug warp-debug.o $(LIBS) -lgsl -lm -lgslcblas

warp.o: warp.cpp
	$(CC) $(CFLAGS) -c warp.cpp $(LIBS) -lgsl -lm -lgslcblas

warp-debug.o: warp-debug.cpp
	$(CC) $(CFLAGS) -c warp-debug.cpp $(LIBS) -lgsl -lm -lgslcblas

clean:
	rm -rf *.o warp warp-debug warp-xiang