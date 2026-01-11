#include <math.h>
#include "io.h"
#include "genrl.h"



void float2rgb(float& value,float& R,float& G,float& B)	//simple color-coding routine
{
   const float dx=0.8;

   value = (6-2*dx)*value+dx;
   R = max(0,(3-fabs(value-4)-fabs(value-5))/2);
   G = max(0,(4-fabs(value-2)-fabs(value-4))/2);
   B = max(0,(3-fabs(value-1)-fabs(value-2))/2);
}


