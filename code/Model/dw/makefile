CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
LIBS = -I/usr/include -L/usr/lib/x86_64-linux-gnu

deepwalk-hete: deepwalk-hete.o
	$(CC) $(CFLAGS) -o deepwalk-hete deepwalk-hete.o $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

deepwalk-hete-em: deepwalk-hete-em.o
	$(CC) $(CFLAGS) -o deepwalk-hete-em deepwalk-hete-em.o $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

deepwalk-hete.o deepwalk-bipa.o: deepwalk-hete.cpp
	$(CC) $(CFLAGS) -c deepwalk-hete.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

deepwalk-hete-em.o: deepwalk-hete.cpp
	$(CC) $(CFLAGS) -c deepwalk-hete-em.cpp $(INCLUDES) $(LIBS) -lgsl -lm -lgslcblas

clean:
	rm -rf *.o deepwalk-hete
