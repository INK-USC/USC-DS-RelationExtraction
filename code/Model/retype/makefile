CC = g++
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
INCLUDES = -I/usr/include -I../eigen-3.2.5
LIBS = -L/usr/lib/x86_64-linux-gnu

retype retype-rm: ransampl.o hplelib.o retype.o retype-rm.o
	$(CC) $(CFLAGS) -o retype-rm ransampl.o hplelib.o retype-rm.o $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas
	$(CC) $(CFLAGS) -o retype ransampl.o hplelib.o retype.o $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

ransampl.o : ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

hplelib.o : hplelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c hplelib.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

retype.o retype-rm.o retype-emtype.o: retype.cpp retype-rm.cpp hplelib.o
	$(CC) $(CFLAGS) -c retype.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas
	$(CC) $(CFLAGS) -c retype-rm.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

clean :
	rm -rf *.o retype retype-rm
