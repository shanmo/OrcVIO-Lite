#pragma once 

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define StateType Rect_<float>

// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{

public:

	KalmanTracker(StateType initRect)
	{
		init_kf(initRect);
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_id = kf_count;
		kf_count++;
		lost_flag = false;
	}

	~KalmanTracker()
	{
		m_history.clear();
	}

	StateType predict();
	void update(StateType stateMat);

	StateType get_state();
	StateType get_rect_xysr(float cx, float cy, float s, float r);

	void update_centroid_history(); 

	static int kf_count;

	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;

    string object_class;
    bool lost_flag;
    vector<Point2f> centroid_history;

private:

	void init_kf(StateType stateMat);

	KalmanFilter kf;
	Mat measurement;

	vector<StateType> m_history;
};

