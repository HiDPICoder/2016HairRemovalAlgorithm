#include <math.h>
#include "color.h"

RGB::RGB()
{
	r = g = b = 0.0f;
}

RGB::RGB(float _r, float _g, float _b)
  : r(_r), g(_g), b(_b)
{ }

HSV::HSV()
{
	h = s = v = 0.0f;
}

HSV::HSV(float _h, float _s, float _v)
  : h(_h), s(_s), v(_v)
{ }

HSV RGB::toHSV()
{
	HSV hsv;

	double min = fmin(r, fmin(g, b));
	double max = fmax(r, fmax(g, b));
	double delta = max - min;

	hsv.v = max;
	hsv.s = max > 1e-6 ? delta/max : 0.0f;

	if (hsv.s > 0.0f)
	{
		if (r == max)       hsv.h = 0.0f + (g-b)/delta;
		else if (g == max)  hsv.h = 2.0f + (b-r)/delta;
		else                hsv.h = 4.0f + (r-g)/delta;

		// Normalize to range [0,1]
		hsv.h /= 6.0f;

		if (hsv.h < 0.0f)   hsv.h += 1.0f;
	}

	return hsv;
}

RGB HSV::toRGB()
{
	if (s == 0.0) return RGB(v, v, v);

	int hi = int(h * 6.0f);
	double f = h * 6.0f - hi;
	double p = v * (1.0f - s);
	double q = v * (1.0f - f * s);
	double t = v * (1.0f - (1.0f - f) * s);

	switch (hi)
	{
		case 0:
		case 6: return RGB(v, t, p);
		case 1: return RGB(q, v, p);
		case 2: return RGB(p, v, t);
		case 3: return RGB(p, q, v);
		case 4: return RGB(t, p, v);
		case 5: return RGB(v, p, q);
	}

	return RGB();
}
