#ifndef OBJ_STATE_H
#define OBJ_STATE_H

#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <Eigen/Eigen>

namespace orcvio 
{

struct ObjectState {

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int object_id;
    std::string object_class;

    Eigen::Vector3d ellipsoid_shape;
    Eigen::Matrix4d object_pose;
    Eigen::MatrixX3d object_keypoints_shape_global_frame;

    /// Default constructor
    ObjectState() {
        
    }

};

/**
 * @brief transform the mean shape to global frame using object pose 
 * @param object_keypoints_mean_shape size 12x3, the mean shape of keypoints 
 * @param object_pose size 4x4, the object pose from object frame to global frame
 * @return size 12x3, object keypoints shape in global frame
 */
Eigen::MatrixX3d transform_mean_keypoints_to_global(const Eigen::MatrixX3d& object_keypoints_mean_shape, const Eigen::Matrix4d& object_pose);

/**
 * @brief save the object state to file  
 * @param an ObjectState structure that holds the object states
 * @param the timestamps of observations 
 * @param the file name to save the results   
 */
void save_object_state_to_file(const ObjectState & object_state, const std::vector<double>& timestamps,
    std::string filepath_format =  "/home/erl/moshan/open_orcvio/catkin_ws_openvins/src/open_vins/results/object_state_%d.txt"  );

}

#endif /* OBJ_STATE_H */
