CC = g++
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result

OBJS = BaseComponentModel.o	EmbeddingModel.o  FctCoarseModel.o	FctDeepModel.o  FullFctModel.o  Instances.o FctConvolutionModel.o FeatureModel.o

all: RE_FCT RE_FCT_fixed

%.o : %.cpp
	$(CC) -c $< -o $@ $(CFLAGS)

RE_FCT : RE_FCT.cpp $(OBJS)
	$(CC) RE_FCT.cpp $(OBJS) -o RE_FCT $(CFLAGS)

RE_FCT_fixed : RE_FCT_fixed.cpp $(OBJS)
	$(CC) RE_FCT_fixed.cpp $(OBJS) -o RE_FCT_fixed $(CFLAGS)

clean:
	rm -rf RE_FCT RE_FCT_fixed *.o
