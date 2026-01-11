/**
 * Connected Components Algorithm
 * Author: Ali Rahimi
 *
 * Source: http://alumni.media.mit.edu/~rahimi/connected/connected.h
 *
 * Modified by Joost Koehoorn, June 2014
 */
#ifndef _CONNECTED_H
#define _CONNECTED_H


#include <vector>
#include <algorithm>
#include <genrl.h>

template <class T, T V>
struct constant
{
	operator T() const { return V; }
};

// HACK
// This does not belong here but we need it in the CComponent struct below.
// Preferably CComponent is just a vector of pixels which is copied over to a
// richer structure with its attributes, but this avoids copying the pixel vector.
struct SBranch
{
	enum State { BLANK, JUNCTION, VISITED };
	std::vector<Coord> pixels;
};
// END HACK

struct CComponent
{
	float boundaryLength;
	unsigned long maxDistance;
	unsigned long skelPixels;
	std::vector<int> pixels;
	std::vector<Coord> junctions;
	std::vector<SBranch> branches;

	CComponent()
	 : boundaryLength(0.0), maxDistance(0), skelPixels(0)
	{  }
};


class ConnectedComponents
{
public:
	ConnectedComponents() {
		clear();
	}
	void clear() {
		labels.clear();
		highest_label = 0;
	}
	template<class Tin, class Tlabel, class Comparator, class Background, class Boolean>
	int connected(const Tin *img, Tlabel *out, std::vector<CComponent>& components,
			 int xM, int yM, int size, Comparator, Background,
		  Boolean K8_connectivity);



private:
	struct Similarity {
		Similarity() : id(0), sameas(0) {}
		Similarity(int _id, int _idx) : id(_id), sameas(_id), idx(_idx) {}
		int id, sameas, tag, idx;
	};

	bool is_root_label(int id) {
		return (labels[id].sameas == id);
	}
	int root_of(int id) {
		while (!is_root_label(id)) {
			// link this node to its parent's parent, just to shorten the tree.
			labels[id].sameas = labels[labels[id].sameas].sameas;

			id = labels[id].sameas;
		}
		return id;
	}
	bool merge(int id1, int id2) {
		int r1 = root_of(id1), r2 = root_of(id2);
		if (r1 != r2) {
			labels[r1].sameas = r2;
			return false;
		}
		return true;
	}
	int new_label(int idx) {
		labels.push_back(Similarity(highest_label, idx));
		return highest_label++;
	}


	template<class Tin, class Tlabel, class Comparator, class Boolean>
	void label_image(const Tin *img, Tlabel *out,
			 int xM, int yM, int size, Comparator,
			 Boolean K8_connectivity);

	template<class Tin, class Tlabel, class Background>
	int relabel_image(const Tin *img, Tlabel *out, std::vector<CComponent>& components, int xM, int yM, int size, Background);


	std::vector<Similarity> labels;
	int highest_label;
};

template<class Tin, class Tlabel, class Comparator, class Background, class Boolean>
int
ConnectedComponents::connected(const Tin *img, Tlabel *labelimg, std::vector<CComponent>& components,
				   int xM, int yM, int size, Comparator SAME, Background IS_BACKGROUND,
				   Boolean K8_connectivity)
{
	label_image(img,labelimg, xM,yM, size, SAME, K8_connectivity);
	return relabel_image(img,labelimg, components, xM,yM, size, IS_BACKGROUND);
}




template<class Tin, class Tlabel, class Comparator, class Boolean>
void
ConnectedComponents::label_image(const Tin *img, Tlabel *labelimg,
				 int xM, int yM, int size, Comparator SAME,
				 const Boolean K8_CONNECTIVITY)
{
	const Tin *row = img;
	const Tin *last_row = 0;
	struct Label_handler {
		Label_handler(const Tin *img, Tlabel *limg) :
			piximg(img), labelimg(limg) {}
		Tlabel &operator()(const Tin *pixp) { return labelimg[pixp-piximg]; }
		const Tin *piximg;
		Tlabel *labelimg;
	} label(img, labelimg);

	clear();

	label(&row[0]) = new_label(0);

	// label the first row.
	for (int c=1, r=0; c<xM; ++c) {
		if (SAME(row[c], row[c-1]))
			label(&row[c]) = label(&row[c-1]);
		else
			label(&row[c]) = new_label(size * r + c);
	}

	// label subsequent rows.
	for (int r=1; r<yM; ++r) {
		// label the first pixel on this row.
		last_row = row;
		row = &img[size*r];

		if (SAME(row[0], last_row[0]))
			label(&row[0]) = label(&last_row[0]);
		else
			label(&row[0]) = new_label(size * r);

		// label subsequent pixels on this row.
		for (int c=1; c<xM; ++c) {
			int mylab = -1;

			// inherit label from pixel on the left if we're in the same blob.
			if (SAME(row[c],row[c-1]))
				mylab = label(&row[c-1]);
			for (int d=(K8_CONNECTIVITY?-1:0); d<1; ++d) {
				// if we're in the same blob, inherit value from above pixel.
				// if we've already been assigned, merge its label with ours.
				if (SAME(row[c], last_row[c+d])) {
					if (mylab>=0) merge(mylab, label(&last_row[c+d]));
					else mylab = label(&last_row[c+d]);
				}
			}
			if (mylab>=0) label(&row[c]) = static_cast<Tlabel>(mylab);
			else label(&row[c]) = new_label(size * r + c);

			if (K8_CONNECTIVITY && SAME(row[c-1], last_row[c]))
				merge(label(&row[c-1]), label(&last_row[c]));
		}
	}
}

template<class Tin, class Tlabel, class Background>
int
ConnectedComponents::relabel_image(const Tin *img, Tlabel *labelimg, std::vector<CComponent>& components, int xM, int yM, int size, Background IS_BACKGROUND)
{
	components.clear();

	int newtag = 0;
	for (int id=0; id<labels.size(); ++id) {
		Similarity& l = labels[id];
		if (!IS_BACKGROUND(img[l.idx]) && is_root_label(id)) {
			components.push_back(CComponent());
			l.tag = newtag++;
		}
	}

	for (int j = 0; j<yM; ++j)
	for (int i = 0; i<xM; ++i) {
		int p = size * j + i;
		if (!IS_BACKGROUND(img[p])) {
			int tag = labels[root_of(labelimg[p])].tag;
			components[tag].pixels.push_back(p);
			labelimg[p] = tag;
		} else {
			labelimg[p] = 0;
		}
	}

	return newtag;
}


#endif // _CONNECTED_H
