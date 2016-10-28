CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
INCLUDES = -I/usr/include -Icode/Model/eigen-3.2.5
LIBS = -L/usr/lib/x86_64-linux-gnu

pte-hete line: ransampl.o linelib.o pte-hete.o line.o
	$(CC) $(CFLAGS) -o pte-hete ransampl.o linelib.o pte-hete.o $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas
	$(CC) $(CFLAGS) -o line ransampl.o linelib.o line.o $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

ransampl.o: ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

linelib.o: linelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c linelib.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

pte-hete.o line.o: pte-hete.cpp line.cpp linelib.o
	$(CC) $(CFLAGS) -c pte-hete.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas
	$(CC) $(CFLAGS) -c line.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

clean:
	rm -rf *.o pte-hete line
