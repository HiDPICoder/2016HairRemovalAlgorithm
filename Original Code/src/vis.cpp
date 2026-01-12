#if GUI_SUPPORT

#include "include/vis.h"
#include "include/skelft.h"
#include <iostream>
#include <GL/glui.h>
#include <math.h>
#include "api.h"


using namespace std;


Display* Display::instance = 0;



Display::Display(IMAGE<float>* _colorImage, IMAGE<float>* _inpainted, FIELD<float>* _mask, FIELD<float>* _skeleton, Options* _options, int argc, char** argv)
  : colorImage(_colorImage), inpainted(_inpainted), mask(_mask), skeleton(_skeleton), options(_options), currentImage(1)
{
	instance = this;

	// Keep track of changes in options
	memcpy(&currentOptions, options, sizeof(Options));

	glutInit(&argc, argv);
	glutInitWindowSize(colorImage->dimX(), colorImage->dimY());
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	mainWindow = glutCreateWindow("Hair detection");
	glutDisplayFunc(displayRelay);
	GLUI_Master.set_glutKeyboardFunc(keyboardRelay);
	GLUI_Master.set_glutReshapeFunc(reshapeRelay);
	GLUI_Master.set_glutIdleFunc(NULL);

	glGenTextures(1, &texture);
	updateTexture();

}

void Display::show()
{
	buildGUI();

	glutMainLoop();
}

void Display::process()
{
	// Do a complete recalculation when using both formats or when a phase 1 option changed.
	// This is necessary because intermediary state is overridden during the processing
	// of the inverted (second format) image. Hence we have to start all over.
	if (
		(options->format != Options::ORIGINAL && options->format != Options::INVERTED) ||
		options->lambda != currentOptions.lambda ||
		options->morphRadius != currentOptions.morphRadius ||
		options->skeletonLevel != currentOptions.skeletonLevel
	)
	{
		computeHairMask();
	}
	else
	{
		computeHairMaskOnlyPhase2();
	}

	memcpy(&currentOptions, options, sizeof(Options));

	updateTexture();
	glutSetWindow(mainWindow);
	glutPostRedisplay();
}

void Display::inpaint()
{
	inpaintImage();

	memcpy(&currentOptions, options, sizeof(Options));

	updateTexture();
	glutSetWindow(mainWindow);
	glutPostRedisplay();
}

void Display::processInpaint()
{
	process();
	inpaint();
}

void updateView(int i)
{
	Display::instance->updateTexture();
	glutSetWindow(Display::instance->mainWindow);
	glutPostRedisplay();
}

void Display::buildGUI()
{
	// view = GLUI_Master.create_glui_subwindow(mainWindow, GLUI_SUBWINDOW_RIGHT);
	view = GLUI_Master.create_glui("Options", 0, colorImage->dimX() + 60);
	view->set_main_gfx_window(mainWindow);

	// SELECT VIEW
	GLUI_Panel *viewPanel = view->add_panel("Image");
	GLUI_RadioGroup *viewRadioGroup = view->add_radiogroup_to_panel(viewPanel, &currentImage, 0, updateView);
	view->add_radiobutton_to_group(viewRadioGroup, "Original");
	view->add_radiobutton_to_group(viewRadioGroup, "Inpainted");
	view->add_radiobutton_to_group(viewRadioGroup, "Hair Mask");
	view->add_radiobutton_to_group(viewRadioGroup, "Skeleton");

	// GAP DETECTION

	GLUI_Panel *gapDetectionPanel = view->add_panel("Gap detection", true);
	gapDetectionPanel->set_alignment(GLUI_ALIGN_LEFT);

	GLUI_Spinner *lambda = view->add_spinner_to_panel(gapDetectionPanel, "Lambda: ", GLUI_SPINNER_FLOAT, &options->lambda);
	lambda->set_float_limits(0.0, 1.0, GLUI_LIMIT_CLAMP);
	lambda->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *skLev = view->add_spinner_to_panel(gapDetectionPanel, "Skeleton level: ", GLUI_SPINNER_FLOAT, &options->skeletonLevel);
	skLev->set_float_limits(2.0, 100.0, GLUI_LIMIT_CLAMP);
	skLev->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *morphRadius = view->add_spinner_to_panel(gapDetectionPanel, "Morph radius: ", GLUI_SPINNER_FLOAT, &options->morphRadius);
	morphRadius->set_float_limits(1.0, 100.0, GLUI_LIMIT_CLAMP);
	morphRadius->set_alignment(GLUI_ALIGN_RIGHT);

	// FILTERING

	GLUI_Panel *filteringPanel = view->add_panel("Filtering", true);
	filteringPanel->set_alignment(GLUI_ALIGN_LEFT);

	view->add_checkbox_to_panel(filteringPanel, "Enabled", &options->filteringEnabled);

	GLUI_Spinner *boundaryPercentage = view->add_spinner_to_panel(filteringPanel, "Boundary %: ", GLUI_SPINNER_FLOAT, &options->boundaryPercentage);
	boundaryPercentage->set_float_limits(0.0, 1.0, GLUI_LIMIT_CLAMP);
	boundaryPercentage->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *minSkeletonLevel = view->add_spinner_to_panel(filteringPanel, "Min skel.lev.: ", GLUI_SPINNER_FLOAT, &options->minSkeletonLevel);
	minSkeletonLevel->set_float_limits(1.0, 100.0, GLUI_LIMIT_CLAMP);
	minSkeletonLevel->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *maxSkeletonLevel = view->add_spinner_to_panel(filteringPanel, "Max skel.lev.: ", GLUI_SPINNER_FLOAT, &options->maxSkeletonLevel);
	maxSkeletonLevel->set_float_limits(1.0, 1000.0, GLUI_LIMIT_CLAMP);
	maxSkeletonLevel->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *minDistanceThreshold = view->add_spinner_to_panel(filteringPanel, "Min distance: ", GLUI_SPINNER_FLOAT, &options->minDistanceThreshold);
	minDistanceThreshold->set_float_limits(1.0, 500.0, GLUI_LIMIT_CLAMP);
	minDistanceThreshold->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *maxDistanceThreshold = view->add_spinner_to_panel(filteringPanel, "Max distance: ", GLUI_SPINNER_FLOAT, &options->maxDistanceThreshold);
	maxDistanceThreshold->set_float_limits(1.0, 500.0, GLUI_LIMIT_CLAMP);
	maxDistanceThreshold->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *maxDistanceScaling = view->add_spinner_to_panel(filteringPanel, "Max distance %: ", GLUI_SPINNER_FLOAT, &options->maxDistanceScaling);
	maxDistanceScaling->set_float_limits(0.0, 1.0, GLUI_LIMIT_CLAMP);
	maxDistanceScaling->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *junctionRatioThreshold = view->add_spinner_to_panel(filteringPanel, "Junction ratio: ", GLUI_SPINNER_FLOAT, &options->junctionRatioThreshold);
	junctionRatioThreshold->set_float_limits(0.0, 1.0, GLUI_LIMIT_CLAMP);
	junctionRatioThreshold->set_alignment(GLUI_ALIGN_RIGHT);

	// BUTTONS

	view->add_statictext("");
	view->add_button("Process", 0, processRelay);
	view->add_button("Inpaint", 0, inpaintRelay);
	view->add_button("Process+Inpaint", 0, processInpaintRelay);
	view->add_statictext("");
	view->add_button("Quit", 0, exit);
	view->add_statictext("");
}



void Display::display()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glBindTexture(GL_TEXTURE_2D, texture);

	glColor3f(1,1,1);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 0.0);
	glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 0.0);
	glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
	glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 1.0);
	glEnd();
	glDisable(GL_TEXTURE_2D);

	glutSwapBuffers();
}

void Display::reshape(int w, int h)
{
	GLUI_Master.auto_set_viewport();

	if (w != colorImage->dimX() || h != colorImage->dimY())
	{
		glutReshapeWindow(colorImage->dimX(), colorImage->dimY());
	}

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Display::updateTexture()
{
	switch (currentImage)
	{
		case 0: generateTexture(colorImage); break;
		case 1: generateTexture(inpainted); break;
		case 2: generateTexture(mask); break;
		case 3: generateTexture(skeleton); break;
	}
}

void Display::generateTexture(FIELD<float>* image)
{
	int dx = image->dimX(), dy = image->dimY();
	unsigned char* tex = new unsigned char[dx * dy * 3];

	int idx = 0;
	for (float* v=image->data(), *vend=v+dx*dy; v<vend; ++v)
	{
		tex[idx++] = *v*0xff;
		tex[idx++] = *v*0xff;
		tex[idx++] = *v*0xff;
	}

	// Create the texture; the texture-id is already allocated
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dx, dy, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);

	delete[] tex;
}

void Display::generateTexture(IMAGE<float>* image)
{
	int dx = image->dimX(), dy = image->dimY();
	unsigned char* tex = new unsigned char[dx * dy * 3];

	int idx = 0;
	for (float* r=image->r.data(), *g=image->g.data(), *b=image->b.data(), *rend=r+dx*dy; r<rend; ++r, ++g, ++b)
	{
		tex[idx++] = *r*0xff;
		tex[idx++] = *g*0xff;
		tex[idx++] = *b*0xff;
	}

	// Create the texture; the texture-id is already allocated
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dx, dy, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);

	delete[] tex;
}

void Display::keyboard(unsigned char k,int,int)
{
	switch (k)
	{
		case ' ':
			if (currentImage++ == 3) currentImage = 0;
			updateTexture();
			break;
		case 27: exit(0);
	}

	glutPostRedisplay();
}

void Display::processRelay(int i) { instance->process(); }
void Display::inpaintRelay(int i) { instance->inpaint(); }
void Display::processInpaintRelay(int i) { instance->processInpaint(); }
void Display::reshapeRelay(int w, int h) { instance->reshape(w, h); }
void Display::displayRelay() { instance->display(); }
void Display::keyboardRelay(unsigned char k, int x, int y) { instance->keyboard(k, x, y); }

#endif
