
EXECUTABLE := dbscan-release

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-std=c++11 -fvisibility=hidden -lpthread -lm -O2 -fopenmp
# Building on Linux
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-std=c++11 -O3 -m64 --gpu-architecture compute_35

OBJS=$(OBJDIR)/main.o $(OBJDIR)/dbscan-seq.o $(OBJDIR)/dbscan-seq2.o $(OBJDIR)/gdbscan.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
