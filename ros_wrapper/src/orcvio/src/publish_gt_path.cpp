#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include "gt_odom.h"

using namespace orcvio;

int main(int argc, char **argv) {

    // Create ros node
    ros::init(argc, argv, "publish_gt_path");
    ros::NodeHandle nh("~");

    // Get parameters to subscribe
    std::string topic;
    nh.getParam("topic_pose", topic);

    // Debug
    ROS_INFO("Done reading config values");
    ROS_INFO(" - topic = %s", topic.c_str());

    OdomToPath gt_odom(nh, topic);

    // Done!
    ros::spin();
    return EXIT_SUCCESS;

}