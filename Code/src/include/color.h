#ifndef _H_COLOR_H_
#define _H_COLOR_H_

class HSV;

class RGB
{
public:
	float r, g, b;

	RGB();
	RGB(float r, float g, float b);

	HSV toHSV();
};

class HSV
{
public:
	float h, s, v;

	HSV();
	HSV(float h, float s, float v);

	RGB toRGB();
};

#endif
