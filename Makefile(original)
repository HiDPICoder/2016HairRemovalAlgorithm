# Optional: Enable GUI (requires GLUI built and paths adjusted)
GUI_SUPPORT = 1

# Detect mpirun for MPI support
HAS_MPIRUN = $(shell which mpirun > /dev/null; echo $$?)

ifeq ($(HAS_MPIRUN), 0)
  MPI_SUPPORT = 1
else
  MPI_SUPPORT = 0
endif

# ---------------------------------------------------------------------
# CUDA detection (system-installed CUDA)
# ---------------------------------------------------------------------

NVCC ?= $(shell which nvcc)
ifeq ($(NVCC),)
  $(error nvcc not found. Please install CUDA Toolkit.)
endif

# For distro-installed CUDA, headers are in /usr/include
CUDAROOT ?= /usr

# ---------------------------------------------------------------------
# Compiler selection
# ---------------------------------------------------------------------

ifeq ($(MPI_SUPPORT), 1)
  CXX = mpic++
else
  CXX = g++
endif

# ---------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------

CFLAGS    = -g -D MPI_SUPPORT=$(MPI_SUPPORT) -D GUI_SUPPORT=$(GUI_SUPPORT) \
            -Wno-deprecated-declarations -ffast-math

CCFLAGS   = -I. -I./src/include -I$(CUDAROOT)/include -O3 -g

# MX250 = Pascal = sm_61 (this is correct for your GPU)
CUDAFLAGS = -use_fast_math \
            -gencode arch=compute_61,code=sm_61 \
            --ptxas-options --maxrregcount=20

# System CUDA libraries (no hardcoded lib64 path)
LIBCUDA   = -lcudart

FRAMEWORKS = -lglut -lGLU -lGL $(LIBCUDA)

# GUI support (optional)
ifeq ($(GUI_SUPPORT),1)
  CCFLAGS    += -I./src/glui/include
  FRAMEWORKS += -lglui
  LINKFLAGS  += -L./src/glui/lib/linux
endif

# ---------------------------------------------------------------------
# Build targets
# ---------------------------------------------------------------------

VPATH        = obj
EXECUTABLE   = hairrazor
BATCH        = batch
BATCH_SRC    = src/batch.cpp

CC_SRC   = $(filter-out $(BATCH_SRC),$(wildcard src/*.cpp))
OBJECTS  = $(subst src/,$(VPATH)/,$(CC_SRC:.cpp=.o)) $(VPATH)/skelft.o
CUDAFILE = src/gaps.cu

all: $(EXECUTABLE) $(BATCH)

$(BATCH): $(BATCH_SRC)
	$(NVCC) -D MPI_SUPPORT=$(MPI_SUPPORT) $(CCFLAGS) $(CUDAFLAGS) -o $@ $< $(LIBCUDA)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) -o $(EXECUTABLE) $(OBJECTS) $(CCFLAGS) $(LINKFLAGS) $(FRAMEWORKS)

$(VPATH)/skelft.o: $(CUDAFILE)
	$(NVCC) -c $(CCFLAGS) $(CUDAFLAGS) -o $@ $(CUDAFILE)

$(VPATH)/%.o: src/%.cpp $(VPATH)/.sentinel
	$(CXX) -MMD $(CFLAGS) $(CCFLAGS) -c -o $@ $<

$(VPATH)/.sentinel:
	mkdir -p $(dir $(OBJECTS))
	touch $@

clean:
	-rm -rf output $(VPATH) $(EXECUTABLE) $(BATCH)

DEPS := $(OBJECTS:.o=.d)
-include $(DEPS)

