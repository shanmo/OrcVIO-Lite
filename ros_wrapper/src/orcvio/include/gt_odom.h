#ifndef GTODOM_H
#define GTODOM_H

#include <string>
#include <fstream>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>

// ref https://gist.github.com/kartikmohta/67cafc968ba5146bc6dcaf721e61b914

namespace orcvio  
{

    /**
     * @brief This class takes in published gt odometry and converts it to path for rviz 
     */
    class OdomToPath {

        public:

        /**
         * @brief Default constructor 
         */
        OdomToPath(ros::NodeHandle& nh, std::string topic) {

            sub_gt_pose = nh.subscribe(topic, 9999, &OdomToPath::gt_odom_path_cb, this);
            pub_gt_path = nh.advertise<nav_msgs::Path>("/orcvio/gt_path", 2);

            first_pose_flag = true;

            set_first_pose_as_origin_flag = true;

            nh.param<std::string>("output_dir_traj", output_dir_traj, "./cache");
            fStateToSave.open((output_dir_traj+"/stamped_groundtruth.txt").c_str(), std::ofstream::trunc);

        };

        /**
         * @brief Default desctructor  
         */
        ~OdomToPath()
        {
            fStateToSave.close();
        }

        /**
         * @brief callback function to convert odometry to path 
         * @param a pointer to the gt odometry 
         */
        void gt_odom_path_cb(const geometry_msgs::PoseStamped::ConstPtr &gt_odom_ptr);

        nav_msgs::Path path;

        bool first_pose_flag;
        bool set_first_pose_as_origin_flag; 
        
        Eigen::Matrix3d R0;
        Eigen::Vector3d p0;

        ros::Subscriber sub_gt_pose;
        ros::Publisher pub_gt_path;

        std::string output_dir_traj;
        std::ofstream fStateToSave;

    };

}

#endif