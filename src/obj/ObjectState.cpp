#include <vector>
#include <iostream>
#include <unordered_map>
#include <Eigen/Eigen>
#include <fstream>
#include <cstdio>
#include <boost/format.hpp>

#include "orcvio/obj/ObjectState.h"

namespace orcvio 
{

Eigen::MatrixX3d transform_mean_keypoints_to_global(const Eigen::MatrixX3d& object_keypoints_mean_shape, const Eigen::Matrix4d& object_pose)
{

    // get the number of keypoints for this object class 
    const int kps_num = object_keypoints_mean_shape.rows();

    Eigen::MatrixX3d object_keypoints_shape_global_frame;
    object_keypoints_shape_global_frame = Eigen::MatrixXd::Zero(kps_num, 3);

    Eigen::Matrix3d wRq = object_pose.block(0, 0, 3, 3);
    Eigen::Vector3d wPq = object_pose.block(0, 3, 3, 1);

    // std::cout << "wRq " << wRq << std::endl;
    // std::cout << "wPq " << wPq.transpose() << std::endl;

    for (int i = 0; i < kps_num; ++i)
    {
        // std::cout << "keypoint position in object frame " << object_keypoints_mean_shape.row(i) << std::endl;
        Eigen::Vector3d keypoint_global_frame = wRq * object_keypoints_mean_shape.row(i).transpose() + wPq;
        // std::cout << "keypoint position in global frame " << keypoint_global_frame.transpose() << std::endl;
        object_keypoints_shape_global_frame.row(i) = keypoint_global_frame.transpose();
    }

    return object_keypoints_shape_global_frame;

}

void save_object_state_to_file(const ObjectState & object_state, const std::vector<double>& timestamps, 
    std::string filepath_format)
{
    boost::format boost_filepath_format(filepath_format);
    std::ofstream file((boost_filepath_format % object_state.object_id).str());

    // std::cout << "debug file " << file.is_open() << std::endl;

    if (file.is_open())
    {
        file << "object id:\n" << object_state.object_id << '\n';
        file << "object class:\n" << object_state.object_class << '\n';
        file << "wTq:\n" << object_state.object_pose << '\n';
        file << "keypoints in global frame:\n" << object_state.object_keypoints_shape_global_frame << '\n';
        file << "ellipsoid shape:\n" << object_state.ellipsoid_shape << '\n';
        file << "observation timestamps:\n";
        for (const auto& time: timestamps)
        {
            file << time << " ";
        }
    }
    // else
    //     std::cout << "cannot open file" << std::endl;

}

}
