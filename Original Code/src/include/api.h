#pragma once

struct Options
{
	char *output;
	char *file;
	int verbose;
	int gui;

	// Gap detection settings
	float skeletonLevel;
	float morphRadius;
	float lambda;

	// Filtering settings
	int filteringEnabled;

	float boundaryPercentage;
	float minSkeletonLevel;
	float maxSkeletonLevel;

	float maxDistanceScaling;
	float minDistanceThreshold;
	float maxDistanceThreshold;

	float junctionRatioThreshold;

	enum Format { ORIGINAL, INVERTED, BOTH, LIKELIEST };
	Format format;

	Options() :
		output(NULL),
		file(NULL),
		verbose(false),
		gui(false),

		skeletonLevel(2.0f),
		morphRadius(5.0f),
		lambda(0.2f),

		filteringEnabled(true),
		boundaryPercentage(0.05),
		minSkeletonLevel(3.0),
		maxSkeletonLevel(40.0),
		minDistanceThreshold(20.0f),
		maxDistanceThreshold(30.0f),
		maxDistanceScaling(0.2f),
		junctionRatioThreshold(0.1f),
		format(LIKELIEST)
	{ }
};

void computeHairMask();
void computeHairMaskOnlyPhase2();
void inpaintImage();
