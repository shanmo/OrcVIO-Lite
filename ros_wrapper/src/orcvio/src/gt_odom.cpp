#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include "gt_odom.h"

namespace orcvio 
{

   void OdomToPath::gt_odom_path_cb(const geometry_msgs::PoseStamped::ConstPtr &gt_odom_ptr){

        if (first_pose_flag)
        {
            p0(0,0) = gt_odom_ptr->pose.position.x;
            p0(1,0) = gt_odom_ptr->pose.position.y;

            p0(2,0) = gt_odom_ptr->pose.position.z;
            // convert robot pose to imu pose
            // p0(2,0) = gt_odom_ptr->pose.position.z + 0.5;

            // convert to quaternion
            Eigen::Quaterniond q0;
            q0.x() = gt_odom_ptr->pose.orientation.x;
            q0.y() = gt_odom_ptr->pose.orientation.y;
            q0.z() = gt_odom_ptr->pose.orientation.z;
            q0.w() = gt_odom_ptr->pose.orientation.w;

            // convert to rotation matrix
            R0 = q0.normalized().toRotationMatrix();

            first_pose_flag = false;
        }

        // convert to quaternion
        Eigen::Quaterniond q1;
        q1.x() = gt_odom_ptr->pose.orientation.x;
        q1.y() = gt_odom_ptr->pose.orientation.y;
        q1.z() = gt_odom_ptr->pose.orientation.z;
        q1.w() = gt_odom_ptr->pose.orientation.w;

        // convert to rotation matrix
        Eigen::Matrix3d R1; 
        R1 = q1.normalized().toRotationMatrix();

        // normalize rotation
        Eigen::Matrix3d R1_normalized;
        if (set_first_pose_as_origin_flag)
            // set first pose as origin
            R1_normalized = R0.transpose() * R1;
        else
            R1_normalized = R1;

        // convert to quaternion
        Eigen::Quaterniond q1_normalized = Eigen::Quaterniond(R1_normalized);
        // normalize the quaternion
        q1_normalized = q1_normalized.normalized();

        geometry_msgs::PoseStamped cur_pose;
        cur_pose.header = gt_odom_ptr->header;
        cur_pose.header.frame_id = "/world";

        // set first pose as origin
        Eigen::Vector3d t1, t1_new;
        t1 << gt_odom_ptr->pose.position.x, gt_odom_ptr->pose.position.y, gt_odom_ptr->pose.position.z;
        // convert robot pose to imu pose 
        // t1 << gt_odom_ptr->pose.position.x, gt_odom_ptr->pose.position.y, gt_odom_ptr->pose.position.z + 0.5;
        
        if (set_first_pose_as_origin_flag)
            // set first pose as origin
            t1_new = R0.transpose() * (t1 - p0);
        else
            t1_new = t1;

        cur_pose.pose.position.x = t1_new(0,0);
        cur_pose.pose.position.y = t1_new(1,0);
        cur_pose.pose.position.z = t1_new(2,0);

        cur_pose.pose.orientation.x = q1_normalized.x(); 
        cur_pose.pose.orientation.y = q1_normalized.y(); 
        cur_pose.pose.orientation.z = q1_normalized.z(); 
        cur_pose.pose.orientation.w = q1_normalized.w(); 

        path.header = cur_pose.header;
        path.header.frame_id = "/world";
        path.poses.push_back(cur_pose);

        pub_gt_path.publish(path);

        // save the pose to txt for trajectory evaluation 
        // timestamp tx ty tz qx qy qz qw
        fStateToSave << cur_pose.header.stamp.toSec() << " "
            << cur_pose.pose.position.x << " " << cur_pose.pose.position.y << " " << cur_pose.pose.position.z << " "
            << cur_pose.pose.orientation.x << " " << cur_pose.pose.orientation.y << " " << cur_pose.pose.orientation.z << " " << cur_pose.pose.orientation.w << std::endl; 
             

        return;

    }

}