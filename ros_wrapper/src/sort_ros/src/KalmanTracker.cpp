#include "sort_ros/KalmanTracker.h"

// tracking id relies on this, so we have to reset it in each seq.
int KalmanTracker::kf_count = 0;

// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum = 7;
	int measureNum = 4;
	kf = KalmanFilter(stateNum, measureNum, 0);

    // State Transition Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 0 0 1 0 0 ]
    // [ 0 1 0 0 0 1 0 ]
    // [ 0 0 1 0 0 0 1 ]
    // [ 0 0 0 1 0 0 0 ]
    // [ 0 0 0 0 1 0 0 ]
    // [ 0 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 0 1 ]
	const float dT = 1.0; 
    setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(4) = dT;
    kf.transitionMatrix.at<float>(12) = dT;
    kf.transitionMatrix.at<float>(20) = dT;
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));

    measurement = Mat::zeros(measureNum, 1, CV_32F);
	setIdentity(kf.measurementMatrix);
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-2));

	// give high uncertainty to the unobservable initial velocities
	setIdentity(kf.errorCovPost, Scalar::all(1));

	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}

// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// predict
	// assert(kf.statePost.type() == CV_32F);
	
	Mat p = kf.predict();
	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	update_centroid_history();
	
	m_history.push_back(predictBox);
	return m_history.back();
}

// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat)
{
	m_time_since_update = 0;
	m_history.clear();
	m_hits += 1;
	m_hit_streak += 1;

	// measurement
	measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	measurement.at<float>(2, 0) = stateMat.area();
	measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

	// update
	kf.correct(measurement);

	update_centroid_history();
}

// Return the current state vector
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}

// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	// handle the case when s * r is negative 
	float w = sqrt(abs(s * r));
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}

void KalmanTracker::update_centroid_history()
{
	Mat s = kf.statePost;
	Point2f cur_centroid(s.at<float>(0, 0), s.at<float>(1, 0));
	centroid_history.emplace_back(cur_centroid); 
}
