#include "pgm.h"
#include "include/genrl.h"
#include "include/io.h"

#include <math.h>
#include <stdio.h>
#include <limits>

void writePGM(unsigned char *im, const char *file, int xM, int yM, int size)
{
	FILE* fp = fopen(file,"wb");
	fprintf(fp,"P5 %d %d 255\n",xM,yM);

	const int SIZE = 3000;
	unsigned char buf[SIZE];

	int bb   = 0;
	for (int j=0; j<yM; ++j)
	for (int i=0; i<xM; ++i)
	{
		buf[bb++] = im[i+size*j];
		if (bb==SIZE)
		{ fwrite(buf,sizeof(unsigned char),SIZE,fp); bb=0; }
	}
	if (bb) fwrite(buf,sizeof(unsigned char),bb,fp);
	fclose(fp);
}

void writePPM(float* data, const char* fname, int xM, int yM, int size)
{
   FILE* fp = fopen(fname,"wb");
   if (!fp) return;

	float m = std::numeric_limits<float>::infinity();
	float M = -std::numeric_limits<float>::infinity();
	for (int j=0; j<yM; ++j)
	for (int i=0; i<xM; ++i)
	{
		float v = data[i+size*j];
		if (v < m && v > 0.0) m = v;
		if (v > M) M = v;
	}

	const int SIZE = 3000;
	unsigned char buf[SIZE];
	int bb=0;

	fprintf(fp,"P6 %d %d 255\n",xM,yM);
	for (int j=0; j<yM; ++j)
	for (int i=0; i<xM; ++i)
	{
		float val = data[i+size*j];
		float r,g,b,v = val > 0.0 ? (val-m)/(M-m)*0.4+0.6 : 0.0;
		// v = max(v,0);
		// if (v>M) { r=g=b=1; } else v = min(v,1);
		float2rgb(v,r,g,b);

		buf[bb++] = (unsigned char)(int)(r*255);
		buf[bb++] = (unsigned char)(int)(g*255);
		buf[bb++] = (unsigned char)(int)(b*255);
		if (bb==SIZE)
		{ fwrite(buf,1,SIZE,fp); bb = 0; }
   }
   if (bb) fwrite(buf,1,bb,fp);

   fclose(fp);
}
