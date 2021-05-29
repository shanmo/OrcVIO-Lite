#ifndef OBJ_FEATURE_H
#define OBJ_FEATURE_H

#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <Eigen/Eigen>

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/mat.hpp"

#include <orcvio/feat/kf.h>
#include <orcvio/utils/se3_ops.hpp>

template<typename _Tp, int m, int n>
  cv::Matx<_Tp, m, n> to_cvmatx(std::initializer_list<_Tp> list) {
   assert(list.size() == m*n);
   std::vector<_Tp> values(list);
   cv::Matx<_Tp, m, n> mat(values.data());
   return mat;
}

namespace orcvio 
{

    /**
     * @brief Object feature class used to collect measurements
     *
     */
    class ObjectFeature {

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        int object_id;
        std::string object_class; 
        int sem_kp_num; 

        std::unordered_map<size_t, std::vector<double> > timestamps;
        int min_triangulation_observations_num;

        vector_eigen<Eigen::MatrixX2d> zs;
        vector_eigen<Eigen::Vector4d> zb;

        // for keypoint trackers 
        // key is part id 
        // value is tracker 
        std::unordered_map<int, KalmanFilter*> kp_trackers; 

        // for plotting 
        cv::Matx<uint8_t, 12, 3> track_colors;

        /// Default constructor
        ObjectFeature(int _object_id, std::string _object_class):
         object_id(_object_id), object_class(_object_class)
        {

            // not used for orcvio-lite 
            // // set number of keypoints for different objects 
            // if (object_class == "car")
            // {
            //     sem_kp_num = 12; 
            // }
            // else if (object_class == "door")
            // {
            //     sem_kp_num = 4; 
            // }    
            // else if (object_class == "barrier")
            // {
            //     sem_kp_num = 8; 
            // }    
            // else if (object_class == "barrel")
            // {
            //     sem_kp_num = 8; 
            // }   
            // else if (object_class == "pylon")
            // {
            //     sem_kp_num = 5; 
            // }   
            // else 
            // {
            //     // unknown class 
            //     sem_kp_num = 0;
            // }

            min_triangulation_observations_num = 3;

            /// Colors in BGR for corresponding labels
            track_colors = to_cvmatx<uint8_t, 12, 3>({ 0, 128, 128,
                                                0, 0, 128,
                                                0, 0, 255,
                                                0, 128, 0,
                                                0, 128, 128,
                                                0, 128, 255,
                                                0, 255, 0,
                                                0, 255, 128,
                                                0, 255, 255,
                                                255, 0, 0,
                                                255, 0, 128,
                                                255, 0, 255 });

        }

        // destructor 
        ~ObjectFeature();

        /**
         * @brief track a semantic keypoint 
         * @param part_id is the part id assigned to semantic keypoint 
         * @param x measurement of keypoint  
         * @param y measurement of keypoint 
         * @param timestamp of the measurement 
         */
        void track_sem_kp(const int & part_id, const float & x, const float & y, const double & timestamp); 

        /**
         * @brief obtain the coordinate of a keypoint  
         * @param part_id is the part id assigned to semantic keypoint 
         * @return coordinate in float type  
         */
        Eigen::Vector2f obtain_kp_coord(const int & part_id);

        /**
         * @brief convert zs to uv norm 
         * @param part_id is the part id that we want to get uvs 
         * @param pointer to a vector to hold keypoint ids that have valid observations 
         * @return flag of whether the observations are sufficient for triangulation
         */
        bool zs_to_uvnorm(const int & part_id, std::unordered_map<size_t, std::vector<Eigen::VectorXd>>& uvs_norm, std::vector<int>& valid_ids);

        /**
         * @brief remove the old observations out of current window 
         * @param valid_times are timestamps that are in current window
         */
        void clean_old_measurements(const std::vector<double>& valid_times);

        /**
         * @brief remove the old boudning box observations out of current window 
         * @param valid_times are timestamps that are in current window
         */
        void clean_old_measurements_lite(const std::vector<double>& valid_times);

        /**
         * @brief get the timestamps and camera poses for the valid observations
         * @param valid ids are ids that have valid observations
         * @param valid_timestamps is a vector to hold the valid timestamps 
         */
        void get_valid_timestamps(const std::vector<int>& valid_ids, std::unordered_map<size_t, std::vector<double>>& valid_timestamps);

        /**
         * @brief draw keypoint tracks on the image 
         * @param the image to draw 
         */
        void draw_kp_track(cv::Mat& img);

        /**
         * @brief get the color of the keypoint track 
         * @param keypoint id 
         */
        cv::Scalar get_kp_track_color(const int& part_id);

    };

}

#endif /* OBJ_FEATURE_H */
