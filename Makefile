# CXX=mpic++
OUTPUTDIR := bin/

CFLAGS := -std=c++11 -fvisibility=hidden -lpthread -lm

ifeq (,$(CONFIG))
	CONFIG := release
endif

ifeq (debug,$(CONFIG))
CFLAGS += -g   # debug flag, no omp should work as well
else
CFLAGS += -O2 -fopenmp 
endif

SOURCES := src/*.cpp
HEADERS := src/*.h
TARGETBIN := dbscan-$(CONFIG)

.PHONY: all clean
all: $(TARGETBIN)

clean:
	/bin/rm -rf $(OBJDIR) *~ $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)
