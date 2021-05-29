#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// --------------------------------------------------------------------
// Kalman Filter Demonstrating, a 2-d ball demo
// --------------------------------------------------------------------

const int winHeight = 600;
const int winWidth = 800;

Point mousePosition = Point(winWidth >> 1, winHeight >> 1);

// mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_MOUSEMOVE) {
		mousePosition = Point(x, y);
	}
}

void TestKF();

int main()
{
	TestKF();

    return 0;
}

void TestKF()
{

    int stateNum = 4;
    int measureNum = 2;
    KalmanFilter kf = KalmanFilter(stateNum, measureNum, 0);

    // initialization
    Mat processNoise(stateNum, 1, CV_32F);
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

	// kf.transitionMatrix = *(Mat_<float>(stateNum, stateNum) <<
	// 	1, 0, 1, 0,
	// 	0, 1, 0, 1,
	// 	0, 0, 1, 0,
	// 	0, 0, 0, 1);
    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(2) = 1.0f;
    kf.transitionMatrix.at<float>(7) = 1.0f;

	setIdentity(kf.measurementMatrix);
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kf.errorCovPost, Scalar::all(1));

    randn(kf.statePost, Scalar::all(0), Scalar::all(winHeight));

	namedWindow("Kalman");
	setMouseCallback("Kalman", mouseEvent);
	Mat img(winHeight, winWidth, CV_8UC3);

	while (1)
	{
		// predict
		Mat prediction = kf.predict();
		Point predictPt = Point(prediction.at<float>(0, 0), prediction.at<float>(1, 0));
		// generate measurement
		Point statePt = mousePosition;
		measurement.at<float>(0, 0) = statePt.x;
		measurement.at<float>(1, 0) = statePt.y;
		// update
		kf.correct(measurement);
		// visualization
		img.setTo(Scalar(255, 255, 255));
		circle(img, predictPt, 8, CV_RGB(0, 255, 0), -1); // predicted point as green
		circle(img, statePt, 8, CV_RGB(255, 0, 0), -1); // current position as red
		imshow("Kalman", img);
		char code = (char)waitKey(100);
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}
	destroyWindow("Kalman");
}

