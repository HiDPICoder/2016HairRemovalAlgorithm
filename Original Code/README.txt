Digital Hair Removal by Threshold Decomposition

(c) Joost Koehoorn, Alex Telea, Univ. of Groningen, 2014
====================================================================================================================

This software package implements the digital hair removal algorithm using CUDA and the Parallel Banding Algorithm and AFMM
for fast skeletonization and morphological erosion/dilation.

1. Building
===========

make

The software needs a C++ compiler, OpenGL, the CUDA 1.1 SDK, and GLUT to build. It may be necessary to modify the
makefile and/or do some small-scale #include adaptions to cope with variations on where the GL, GLUT, and CUDA headers
are. The build was tested on Mac OS X 10.9/10.10 with gcc 4.9 and on Linux with gcc 4.6.

The makefile tries to detect if the platform supports MPI by checking if `mpirun` is available in $PATH. To override
this, simply hardcode the value for `MPI_SUPPORT` to either 0 or 1.

A basic GUI to interactively change parameters is provided when GUI_SUPPORT is set to 1. This requires linking with
the GLUI library, of which a Mac OS and Linux64 compiled library is provided, but recompiling GLUI from source may be
necessary on other platforms.

2. Running
==========

Single image
------------

./hairrazor [options]

    Option  Arg     Default  Description
    ------------------------------------------------------------------
    -f      path             Input PPM file, required.
    -w      0..3    3        0 = original, 1 = inverted, 2 = combined, 3 = likeliest.

    -p      [0,1]   0.05     Skeleton prune as percentage from boundary.
    -b      [1,..)  3        Minimum prune parameter.
    -B      [1,..)  40       Maximum prune parameter.

    -s      [0,1]   0.2      Percentage of longest branch to require as minimal branch length.
    -d      [1,..)  20       Minimum bound on branch length.
    -D      [1,..)  30       Maximum bound on branch length.

    -J      [0,1]   0.1      Maximum junction ratio.
    -S      [1,..)  2        Skeletonization simplification level.
    -r      [1,..)  5        Radius used for opening/closings.
    -l      [0,1]   0.2      Lambda, 0 = open-close, 1 = close-open, in-between for linear combination.

    -o      path             Output directory.
    -v                       Verbose, output information.

The input image must be provided in PPM format. The output file is written into ./output/inpainted.ppm alongside various
intermediate images, however using `-o path` a different output path can be given. When compiled with GUI support, provide
`-g` to open a GUI where parameters may be changed interactively.

Batch processing
----------------

./batch [options] -- paths...

A separate program for batch processing is provided. Options to `hairrazor` can be specified by separating the paths by
using two dashes. Output files are written to ./output/{image.ppm}/inpainted.ppm.
