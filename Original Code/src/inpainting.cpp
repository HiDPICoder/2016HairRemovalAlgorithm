#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "field.h"
#include "flags.h"
#include "image.h"
#include "genrl.h"
#include "io.h"
#include "mfmm.h"


using namespace std;

//----------------------------------------------------------------


FIELD<float>* compute_distance(FIELD<float>*,float,float);
void compute_gradient(FIELD<float>*,FIELD<float>*& gx,FIELD<float>*& gy);
void executeInpaint(IMAGE<float>* rgbImage, FIELD<float>* mask);
float length;

FIELD<float>* grad_x,*grad_y;				//the complete (inside/outside) distance-gradient field
FIELD<float>* dist;					//the complete (inside/outside) distance-to-bounday field
FLAGS* flags;						//the FMM flags-field
int   B_radius = 5;					//the inpainting neighborhood radius
int   dst_wt = 1;					//use dist-weighting in inpainting (t/f)
int   lev_wt = 1;					//use level-weighting in inpainting (t/f)
const int   N = 0;					//window for smoothing during gradient computation
							//(if N=0, no smoothing is attempted)

//---------------------------------------------------------------------------------------------------
const float S = 0.8, G = 0.2;
const float S8 = S/8;
const float SG = 3*S/4 - G;
const float S4 = -4*S;

float w[5][5] = {{S8, 0, SG, 0, S8},{0, 0, S4, 0, 0},
{SG, S4, 1+4*G+12.5*S, S4, SG},{0, 0, S4, 0, 0},{S8, 0, SG, 0, S8}};

void inpaint(IMAGE<float>* rgbImage, FIELD<float>* mask)
{
   float k = -1;					//Threshold
   float sk_lev = 20;					//
   int  twopass = 1;					//Using 2-pass or 1-pass method for boundary treatment

   FIELD<float>* f = new FIELD<float>(*mask);

   dist    = compute_distance(f,k,2*B_radius);          //compute complete distance field in a band 2*B_radius around the inpainting zone
   compute_gradient(dist,grad_x,grad_y);                //compute smooth gradient of distance field

   flags = new FLAGS(*f,0.5f);
   executeInpaint(rgbImage, f);

   delete f;
   delete flags;
}



void executeInpaint(IMAGE<float>* rgbImage, FIELD<float>* mask)
{
   int nfail,nextr;
   FLAGS* fl = new FLAGS(*flags); FIELD<float>* ff = new FIELD<float>(*mask);
   ModifiedFastMarchingMethod mfmm(ff,fl,rgbImage,grad_x,grad_y,dist,B_radius,dst_wt,lev_wt,1000000);
   mfmm.execute(nfail,nextr); delete fl; delete ff;
   // rgbImage->normalize();
}


FIELD<float>* compute_distance(FIELD<float>* fi,float k,float maxd)
{
   int nfail,nextr;
   FIELD<float>*    fin = new FIELD<float>(*fi);	//Copy input field
   FLAGS*   	flagsin = new FLAGS(*fin,k);		//Make flags field
   FLAGS*       fcopy   = new FLAGS(*flagsin);          //Copy flags field for combining the two fields afterwards
   FastMarchingMethod fmmi(fin,flagsin);
   fmmi.execute(nfail,nextr);

   FIELD<float>*   fout = new FIELD<float>(*fi);	//Copy input field
   FLAGS*      flagsout = new FLAGS(*fout,-k);		//Make flags field
   FastMarchingMethod fmmo(fout,flagsout);
   fmmo.execute(nfail,nextr,2*B_radius);		//Execute FMM only in a band 2*B_radius deep, we need no more

   FIELD<float>* f = new FIELD<float>(*fin);		//Combine in and out-fields in a single distance field 'f'
   for(int i=0;i<f->dimX();i++)
     for(int j=0;j<f->dimY();j++)
     {
        if (fcopy->alive(i,j)) f->value(i,j) = -fout->value(i,j);
	if (flagsout->faraway(i,j)) f->value(i,j) = 0;
     }

   delete flagsin; delete flagsout; delete fin; delete fout; delete fcopy;
   return f;						//All done, return 'f'
}


void gradient_filter(FIELD<float>* f,int i,int j,float& gx,float& gy)	//compute gradient of f[i][j] in gx,gy
{									//by using smoothing on a N-pixel neighborhood
  gx = gy = 0;  float ci = 0, cj = 0; float wsi = 0, wsj = 0;

  for(int ii=-N;ii<=N;ii++)
      for(int jj=-N;jj<=N;jj++)
      {
        ci += w[N+ii][N+jj]*ii*f->value(i+ii,j+jj);
        cj += w[N+ii][N+jj]*jj*f->value(i+ii,j+jj);
        wsi += w[N+ii][N+jj]*ii*ii;
        wsj += w[N+ii][N+jj]*jj*jj;
      }

  gx = ci/wsi; gy = cj/wsj;                             //normalize gradient
  float r = sqrt(gx*gx+gy*gy);
  gx /= r; gy /= r;
}



void gradient(FIELD<float>* f,int i,int j,float& gx,float& gy)	//compute gradient of f[i][j] in gx,gy
{
  gx = gy = 0;  float ci = 0, cj = 0; float wsi = 0, wsj = 0;

  const int N=0;
  for(int ii=0;ii<=N;ii++)
      for(int jj=-N;jj<=N;jj++)
      {
        ci += f->value(i+ii+1,j+jj)-f->value(i+ii,j+jj);
        cj += f->value(i+ii,j+jj+1)-f->value(i+ii,j+jj);
      }
  const float SZ = 2*N+1;

  gx = ci/SZ; gy = cj/SZ;                             //normalize gradient
  float r = sqrt(gx*gx+gy*gy);
  if (r>0.00001) { gx /= r; gy /= r; }
}



void compute_gradient(FIELD<float>* f,FIELD<float>*& gx,FIELD<float>*& gy)
{								//compute gradient of 'f' in 'gx','gy'
  gx = new FIELD<float>(f->dimX(),f->dimY()); *gx = 0;
  gy = new FIELD<float>(f->dimX(),f->dimY()); *gy = 0;

  int i,j;
  if (N)							//N>0? Use gradient-computation by smoothing
    for(i=N+1;i<f->dimX()-N-1;i++)				//with a filter-size of N pixels
       for(j=N+1;j<f->dimY()-N-1;j++)
          gradient_filter(f,i,j,gx->value(i,j),gy->value(i,j));
  else								//N=0? Use no smoothing, compute gradient directly
     for(i=N+1;i<f->dimX()-N-1;i++)				//by central differences.
       for(j=N+1;j<f->dimY()-N-1;j++)
          gradient(f,i,j,gx->value(i,j),gy->value(i,j));
}

