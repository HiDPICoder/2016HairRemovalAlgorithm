#pragma once

#include <stdlib.h>
#include "field.h"
#include "image.h"
#include "api.h"
#include <GL/glui.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

class	Display
{
public:
	Display(IMAGE<float>* colorImage, IMAGE<float>* inpainted, FIELD<float>* mask, FIELD<float>* skeleton, Options* options, int argc, char** argv);

	void updateTexture();
	void generateTexture(FIELD<float>* image);
	void generateTexture(IMAGE<float>* image);

	IMAGE<float>* colorImage;
	IMAGE<float>* inpainted;
	FIELD<float>* mask;
	FIELD<float>* skeleton;

	GLuint texture;
	GLUI* view;
	Options* options;
	Options currentOptions;
	int mainWindow;
	int currentImage;

	void show();

private:
	void buildGUI();
	void process();
	void inpaint();
	void processInpaint();
	void display();
	void reshape(int w, int h);
	void keyboard(unsigned char k,int,int);

	static void processRelay(int i);
	static void inpaintRelay(int i);
	static void processInpaintRelay(int i);
	static void displayRelay();
	static void reshapeRelay(int w, int h);
	static void keyboardRelay(unsigned char k,int,int);

	static Display* instance;
};
