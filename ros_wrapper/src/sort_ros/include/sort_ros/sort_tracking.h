#pragma once 

#include <iostream>
#include <fstream>
#include <set>
#include <string>

#include "sort_ros/Hungarian.h"
#include "sort_ros/KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

namespace sort_ros {

// global variables for counting colors 
#define CNUM 20

typedef struct TrackingBox
{
	vector<int> frame;
	int id;
	Rect_<float> box;
    string object_class;

}TrackingBox;

struct Config {
    int max_age;
    int min_hits;
    double iou_threshold;
    bool use_centroid_dist_flag; 
    double centroid_dist_threshold; 
};


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);

// Computes centroid distance between two bounding boxes
double GetCentroidDist(Rect_<float> bb_test, Rect_<float> bb_gt);

// Computes the centroid of a bounding box
Point2f GetRectCenter(Rect_<float> bb);

class SortTracker {

public:

    SortTracker() : frame_count(0), max_age(3), min_hits(5), trkNum(0), detNum(0), iou_threshold(0.3)
    {
        gen_class();
        gen_colors();
    }

    void update(vector<TrackingBox> detFrameData);
    void remove_lost_trackers();
    void set_config(Config config);

    void gen_colors();
    Mat draw_bbox(Mat img);
    Mat draw_centroids(Mat img);

    void gen_class();
    bool check_valid_class(string label);

    // we only keep the bbox that is currently tracked 
    // if a bbox gets lost we publish it and remove it from trackers
    vector<KalmanTracker> trackers;

private:

    int frame_count;
    int max_age;
    int min_hits;

    unsigned int trkNum;
    unsigned int detNum;

    double iou_threshold;
    bool use_centroid_dist_flag; 
    double centroid_dist_threshold;

    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> distMatrix;
    vector<int> assignment;

	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;

    vector<Point> matchedPairs;

    Scalar_<int> randColor[CNUM];
    set<string> class_labels; 
};

}


