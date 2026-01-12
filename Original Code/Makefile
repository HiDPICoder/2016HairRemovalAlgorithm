# Enabling GUI support requires linking with GLUI
GUI_SUPPORT = 0
HAS_MPIRUN = $(shell which mpirun > /dev/null; echo $$?)

ifeq ($(HAS_MPIRUN), 0)
  MPI_SUPPORT = 1
else
  MPI_SUPPORT = 0
endif

ifeq ("$(shell uname)", "Linux")
  OS="Linux"
endif
ifeq ("$(shell uname)", "Darwin")
  OS="MacOSX"
endif

ifndef OS
  $(error Unsupported operating system)
endif

NVSDKINC=

# Mac OS X defines
ifeq ($(OS), "MacOSX")
  CUDAROOT  = /usr/local/cuda
  CUDAINC   = "/Developer/GPU Computing/shared/inc"

  ifeq ($(MPI_SUPPORT), 1)
  CXX       = mpic++
  else
  CXX       = c++
  endif

  LIBCUDA   = -L$(CUDAROOT)/lib -lcudart
  FRAMEWORKS = -framework OpenGL -framework GLUT $(LIBCUDA)

  ifeq ($(GUI_SUPPORT),1)
    FRAMEWORKS += -lglui
    LINKFLAGS = -L./src/glui/lib/mac
  endif

# Linux defines
else
  CUDAROOT  = /usr/local/cuda-6.0
  CUDAINC   = $(CUDAROOT)/include

  ifeq ($(MPI_SUPPORT), 1)
  CXX       = /opt/intel/impi/4.1.1.036/intel64/bin/mpicxx
  else
  CXX       = c++
  endif

  LIBCUDA   = -L$(CUDAROOT)/lib64 -lcudart
  FRAMEWORKS = -lGL -lGLU -lglut $(LIBCUDA)

  ifeq ($(GUI_SUPPORT),1)
    FRAMEWORKS += -lglui
    LINKFLAGS = -L./src/glui/lib/linux
  endif
endif



CFLAGS    = -g -D MPI_SUPPORT=$(MPI_SUPPORT) -D GUI_SUPPORT=$(GUI_SUPPORT) -fno-access-control -Wno-deprecated -Wno-deprecated-declarations -Wno-unused-result -ffast-math
CCFLAGS   = -I. -I./src/glui/include -I./src/include -O3 -I/usr/local/cuda/include -g

CUDAFLAGS = -use_fast_math -gencode=arch=compute_11,code=\"sm_11,compute_11\" --ptxas-options= --maxrregcount 20

#--------------------------------------------------------------------------------------------------------------------------------

NVCC = $(CUDAROOT)/bin/nvcc

VPATH   = obj
EXECUTABLE = hairrazor
BATCH = batch
BATCH_SRC = src/batch.cpp

CC_SRC   = $(filter-out $(BATCH_SRC),$(wildcard src/*.cpp))
OBJECTS  = $(subst src/,$(VPATH)/,$(CC_SRC:.cpp=.o)) $(VPATH)/skelft.o
CUDAFILE = src/gaps.cu

all: $(EXECUTABLE) $(BATCH)

$(BATCH): $(BATCH_SRC)
	$(NVCC) -D MPI_SUPPORT=$(MPI_SUPPORT) $(CCFLAGS) $(CUDAFLAGS) $(LIBCUDA) -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	$(CXX) -o $(EXECUTABLE) $(OBJECTS) $(CCFLAGS) $(LINKFLAGS) $(FRAMEWORKS)

$(VPATH)/skelft.o: $(CUDAFILE)
	$(NVCC) -c $(CCFLAGS) -I$(CUDAINC) $(CUDAFLAGS) -o $@ $(CUDAFILE)

$(VPATH)/%.o: src/%.cpp $(VPATH)/.sentinel
	$(CXX) -MMD $(CFLAGS) $(CCFLAGS) $(EXTRAFLAGS) -c -o $@ $<

$(VPATH)/.sentinel:
	mkdir -p $(dir $(OBJECTS))
	touch $@

clean:
	-rm -rf output $(VPATH) $(EXECUTABLE) $(BATCH)


DEPS := $(OBJECTS:.o=.d)

-include $(DEPS)

