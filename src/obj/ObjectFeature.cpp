#include <math.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <unistd.h>

#ifndef NDEBUG
#define NDEBUG false
#endif

constexpr bool DEBUG = (! NDEBUG);

#include "orcvio/obj/ObjectFeature.h"

namespace orcvio 
{

void ObjectFeature::track_sem_kp(const int & part_id, const float & x, const float & y, const double & timestamp)
{

    if(kp_trackers.find(part_id) == kp_trackers.end()) {
        if (x == 0 && y == 0)
        {
            // this should not happen, we think this kp is already initialized
            // but in fact it is not 
            
            // this case may happen when an object is re-detected 
            // ignore this for now 
            return; 
        }
        else 
        {
            /* kp is NOT being tracked */
            kp_trackers[part_id] = new KalmanFilter; 
        }
    } else {
        /* kp is being tracked */
        // pass 
    }

    MeasurementPackage meas_package;
    meas_package.timestamp_ = timestamp;
    meas_package.raw_measurements_ = Eigen::VectorXd(2);
    meas_package.raw_measurements_ << static_cast<double>(x), static_cast<double>(y);

    kp_trackers[part_id]->ProcessMeasurement(meas_package);

} 

Eigen::Vector2f ObjectFeature::obtain_kp_coord(const int & part_id)
{

    Eigen::Vector2f pos; 

    // make sure the part_id is currently being tracked 
    if(kp_trackers.find(part_id) == kp_trackers.end()) 
    {
        pos << 0, 0;
        return pos;
    }

    pos(0) = static_cast<float>(kp_trackers[part_id]->x_(0));
    pos(1) = static_cast<float>(kp_trackers[part_id]->x_(1));

    // keep track of kp histroy for plotting 
    kp_trackers[part_id]->kp_history.push_back(pos);

    // for debugging 
    // std::cout << "kps obtained " << pos << std::endl; 

    return pos; 
}

bool ObjectFeature::zs_to_uvnorm(const int & part_id, std::unordered_map<size_t, std::vector<Eigen::VectorXd>>& uvs_norm, std::vector<int>& valid_ids)
{
    
    // we only keep object observations in one camera
    int cam_id = 0;
    Eigen::Vector2d uv_n;
    int valid_num = 0;

    for (int i = 0; i < static_cast<int>(zs.size()); ++i)
    {
        Eigen::MatrixX2d matrix = zs.at(i);
        
        // for debugging 
        // std::cout << "matrix " << matrix << std::endl;

        // extract the observations for part id
        uv_n = matrix.row(part_id);

        // check whether the observations are invalid
        if (! uv_n.allFinite())
            continue;

        valid_ids.push_back(i);

        // insert valid observations in uvs_norm
        uvs_norm[cam_id].emplace_back(uv_n);

        // std::cout << "uv_n " << uv_n << std::endl;

        ++valid_num;
    }

    if (valid_num > min_triangulation_observations_num)
        return true;
    else 
    {
        // for debugging 
        // std::cout << "valid_num " << valid_num << std::endl;
        // std::cout << "min_triangulation_observations_num " << min_triangulation_observations_num << std::endl;

        return false;
    }
}

void ObjectFeature::get_valid_timestamps(const std::vector<int>& valid_ids, std::unordered_map<size_t, std::vector<double>>& valid_timestamps)
{

    // // for debugging
    // for (const auto & id : valid_ids)
    //     std::cout << "id " << id << std::endl;
    // std::exit(0);

    for (const auto& id : valid_ids)
    {

        // insert valid timestamps
        valid_timestamps[0].emplace_back(timestamps.at(0).at(id));

    }

}

void ObjectFeature::clean_old_measurements(const std::vector<double>& valid_times)
{

     // for debugging
     if (DEBUG) {
         std::ofstream outfile;
         outfile.open("/tmp/clean_old_mesaurements_valid_times_" + std::to_string(getpid()) + ".txt", std::ios_base::app); // append instead of overwrite
         bool first_item = true;
         for (const auto & time : valid_times) {
              outfile << (first_item ? "" : ", ") << time;
              first_item = false;
         }
         outfile << "\n";
     }

    // Loop through each of the cameras we have
    for(auto const &pair : timestamps) {

        // Assert that we have all the parts of a measurement
        assert(timestamps.at(pair.first).size() == zs.size());
        assert(timestamps.at(pair.first).size() == zb.size());

        // Our iterators
        auto it1 = timestamps.at(pair.first).begin();
        auto it2 = zs.begin();
        auto it3 = zb.begin();

        // Loop through measurement times, remove ones that are not in our timestamps
        while (it1 != timestamps.at(pair.first).end()) {
            if (std::find(valid_times.begin(),valid_times.end(), *it1) == valid_times.end()) {
                it1 = timestamps.at(pair.first).erase(it1);
                it2 = zs.erase(it2);
                it3 = zb.erase(it3);
            } else {
                ++it1;
                ++it2;
                ++it3;
            }
        }


    }

}

void ObjectFeature::clean_old_measurements_lite(const std::vector<double>& valid_times)
{

    // Loop through each of the cameras we have
    for(auto const &pair : timestamps) {

        // Assert that we have all the parts of a measurement
        assert(timestamps.at(pair.first).size() == zb.size());

        // Our iterators
        auto it1 = timestamps.at(pair.first).begin();
        auto it3 = zb.begin();

        // Loop through measurement times, remove ones that are not in our timestamps
        while (it1 != timestamps.at(pair.first).end()) {
            if (std::find(valid_times.begin(),valid_times.end(), *it1) == valid_times.end()) {
                it1 = timestamps.at(pair.first).erase(it1);
                it3 = zb.erase(it3);
            } else {
                ++it1;
                ++it3;
            }
        }


    }

}

void ObjectFeature::draw_kp_track(cv::Mat& img)
{
    // int radius = std::max(2, 2 * img.rows / 40);
    int radius = std::max(2, 2 * img.rows / 40) / 2;

    // iterate the trackers 
    for (const auto & tracker : kp_trackers)
    {
        cv::Scalar kp_col = get_kp_track_color(tracker.first);
        // Draw tracked features.
        cv::Point2f prev_pt, curr_pt;
        prev_pt.x = 0;
        prev_pt.y = 0;
        for (const auto& kp : tracker.second->kp_history) {
            curr_pt.x = kp(0);
            curr_pt.y = kp(1); 
            if (prev_pt.x == 0 && prev_pt.y == 0)
            {
                // nothing to plot 
            }
            else 
            {
                // plot the old kp 
                // circle(img, prev_pt, 2, kp_col, 5);
                cv::line(img, prev_pt, curr_pt, kp_col, /*lineThickness=*/3);
            }
            prev_pt = curr_pt;
        }
        // plot the most recent kp 
        circle(img, curr_pt, radius, kp_col, /*lineThickness=*/-.3);
    }
}

cv::Scalar ObjectFeature::get_kp_track_color(const int& part_id) 
{
    if (part_id < 12)
    {
        auto col = track_colors.row(part_id);
        return cv::Scalar(col(0,0), col(0,1), col(0,2));
    }
    else 
    {
        return cv::Scalar(255, 0, 0);
    }
}

ObjectFeature::~ObjectFeature() 
{ 
    // for (auto tracker : kp_trackers)
    //     delete tracker.second; 
} 

} // end namespace orcvio 
