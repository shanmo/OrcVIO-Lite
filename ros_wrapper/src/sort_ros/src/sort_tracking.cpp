#include "sort_ros/sort_tracking.h"

using namespace std;
using namespace cv;

namespace sort_ros
{
    double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
    {
        float in = (bb_test & bb_gt).area();
        float un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return (double)(in / un);
    }

    Point2f GetRectCenter(Rect_<float> bb)
    {
        Point2f center_of_rect = (bb.br() + bb.tl()) * 0.5;
        return center_of_rect;
    }

    double GetCentroidDist(Rect_<float> bb_test, Rect_<float> bb_gt)
    {
        Point2f centroid_test = GetRectCenter(bb_test);
        Point2f centroid_gt = GetRectCenter(bb_gt);

        //calculating Euclidean distance
        double x_diff = centroid_test.x - centroid_gt.x; 
        double y_diff = centroid_test.y - centroid_gt.y;
        double dist = pow(x_diff, 2) + pow(y_diff, 2);   
        dist = sqrt(dist);   

        return dist;
    }

    void SortTracker::remove_lost_trackers()
    {
        for (auto it = trackers.begin(); it != trackers.end();)
        {

            // remove lost trackers
            if (it->lost_flag == true)
                it = trackers.erase(it);
            else
                it++;
        }
    }

    void SortTracker::set_config(Config config)
    {
        max_age = config.max_age;
        min_hits = config.min_hits;
        iou_threshold = config.iou_threshold;
        use_centroid_dist_flag = config.use_centroid_dist_flag;
        centroid_dist_threshold = config.centroid_dist_threshold;
    }

    void SortTracker::update(vector<TrackingBox> detFrameData)
    {
        if (trackers.size() == 0) // the first frame met
        {
            // initialize kalman trackers using first detections.
            for (unsigned int i = 0; i < detFrameData.size(); i++)
            {
                KalmanTracker trk = KalmanTracker(detFrameData[i].box);
                trk.object_class = detFrameData[i].object_class;
                trackers.push_back(trk);
            }

            return;
        }

        ///////////////////////////////////////
        // remove dead trackers
        // we can't put this in the callback
        remove_lost_trackers();

        ///////////////////////////////////////
        // 3.1. get predicted locations from existing trackers.

        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end(); it++)
        {
            Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
            }
            else
            {
                it->lost_flag = true;
                // cerr << "Box invalid at frame: " << frame_count << endl;
            }
        }

        ///////////////////////////////////////
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
        // dets : detFrameData

        trkNum = predictedBoxes.size();
        detNum = detFrameData.size();

        distMatrix.clear();
        distMatrix.resize(trkNum, vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++)
            {
                if (!use_centroid_dist_flag)
                {
                    // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                    distMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[j].box);
                }
                else 
                {
                    // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                    distMatrix[i][j] = GetCentroidDist(predictedBoxes[i], detFrameData[j].box);
                }
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(distMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }

        matchedPairs.clear();

        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) // pass over invalid values
                continue;

            if (!use_centroid_dist_flag)
            {
                // filter out matched with low IOU
                if (1 - distMatrix[i][assignment[i]] < iou_threshold)
                {
                    unmatchedTrajectories.insert(i);
                    unmatchedDetections.insert(assignment[i]);
                }
                else
                    matchedPairs.push_back(Point(i, assignment[i]));
            }
            else 
            {
                // filter out matched with large centroid distance 
                if (distMatrix[i][assignment[i]] > centroid_dist_threshold)
                {
                    unmatchedTrajectories.insert(i);
                    unmatchedDetections.insert(assignment[i]);
                }
                else
                    matchedPairs.push_back(Point(i, assignment[i]));
            }
        }

        ///////////////////////////////////////
        // 3.3. updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker
        int detIdx, trkIdx;

        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detFrameData[detIdx].box);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detFrameData[umd].box);
            tracker.object_class = detFrameData[umd].object_class;
            trackers.push_back(tracker);
        }

        for (auto it = trackers.begin(); it != trackers.end(); it++)
        {
            // remove dead tracklet
            if ((*it).m_time_since_update > max_age)
            {
                it->lost_flag = true;
                // cerr << "Box invalid at frame: " << frame_count << endl;
            }
        }

        // this is not used, since we use image timestamps as frame id
        frame_count++;
        //cout << frame_count << endl;
    }

    void SortTracker::gen_colors()
    {
        // 0. randomly generate colors, only for display
        RNG rng(0xFFFFFFFF);

        for (int i = 0; i < CNUM; i++)
            rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
    }

    Mat SortTracker::draw_bbox(Mat img)
    {
        Mat detection_image = img.clone();

        if (detection_image.channels() == 1)
        {
            // convert to rgb image if necessary
            cv::cvtColor(detection_image, detection_image, cv::COLOR_GRAY2BGR);
        }


        for (auto it = trackers.begin(); it != trackers.end(); it++)
        {
            Scalar_<int> color = randColor[((*it).m_id + 1) % CNUM];
            if (((*it).m_time_since_update == 0) && ((*it).m_hit_streak >= min_hits))
            {
                rectangle(detection_image, (*it).get_state(), color, 2, 8, 0);

                char str[200];
                sprintf(str, "ID:%d", (*it).m_id + 1);
                putText(detection_image, str, Point2f((*it).get_state().x, (*it).get_state().y), FONT_HERSHEY_PLAIN, 2, color, 5);
            }
        }

        return detection_image;
    }

    Mat SortTracker::draw_centroids(Mat img)
    {
        Mat detection_image = img.clone();

        if (detection_image.channels() == 1)
        {
            // convert to rgb image if necessary
            cv::cvtColor(detection_image, detection_image, cv::COLOR_GRAY2BGR);
        }


        for (auto it = trackers.begin(); it != trackers.end(); it++)
        {
            Scalar_<int> color = randColor[((*it).m_id + 1) % CNUM];
            if (((*it).m_time_since_update == 0) && ((*it).m_hit_streak >= min_hits))
            {
                int count = 0; 
                Point2f cur_centroid, prev_centroid; 
                for (const auto & pt : (*it).centroid_history)
                {
                    cur_centroid = pt; 
                    circle(detection_image, pt, 3, color, CV_FILLED, 0); 

                    if (count != 0)
                    {
                        line(detection_image, prev_centroid, cur_centroid, color, 5, LINE_AA, 0);
                    }
                    prev_centroid = cur_centroid;
                    ++count;
                }
                char str[200];
                sprintf(str, "ID:%d", (*it).m_id + 1);
                putText(detection_image, str, (*it).centroid_history.back(), FONT_HERSHEY_PLAIN, 2, randColor[((*it).m_id + 1) % CNUM], 5);
            }
        }

        return detection_image;
    }

    void SortTracker::gen_class()
    {
        // for kitti dataset 
        // class_labels.insert("car");
        // class_labels.insert("truck");
        // class_labels.insert("bus");

        // for tum rgbd dataset 
        // class_labels.insert("laptop");
        // class_labels.insert("book");
        // class_labels.insert("tvmonitor");
        // class_labels.insert("cup");
        // class_labels.insert("keyboard");
        // class_labels.insert("mouse");
        
        // for visma dataset
        // class_labels.insert("chair");

        // for erl 
        class_labels.insert("chair");
        class_labels.insert("tvmonitor");
    }


    bool SortTracker::check_valid_class(string label)
    {
        if (class_labels.find(label) != class_labels.end())
            return true;
        else
            return false;

        return true; 
    }


} // namespace sort_ros
