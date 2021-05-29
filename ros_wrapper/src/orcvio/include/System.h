//
// Managing the image processer and the estimator.
//

#ifndef SYSTEM_H
#define SYSTEM_H

#include <boost/shared_ptr.hpp>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <typeinfo>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <std_msgs/Float64MultiArray.h>
#include <eigen_conversions/eigen_msg.h>

#include <vector>
#include <fstream>
#include <string>
#include <queue>
#include <mutex>

#include <orcvio/image_processor.h>
#include <orcvio/orcvio.h>
#include <orcvio/utils/math_utils.hpp>
#include <orcvio/dataset_reader.h>

#include "sensors/ImuData.hpp"
#include "sensors/ImageData.hpp"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

namespace orcvio {

/*
 * @brief Manager of the system.
 */
class System {

public:

    // Constructor
    System(ros::NodeHandle& n);
    // Disable copy and assign constructors.
    System(const ImageProcessor&) = delete;
    System operator=(const System&) = delete;

    // Destructor.
    ~System();

    // Initialize the object.
    bool initialize();

    typedef boost::shared_ptr<System> Ptr;
    typedef boost::shared_ptr<const System> ConstPtr;

private:

    // Ros node handle.
    ros::NodeHandle nh;

    // Subscribers.
    ros::Subscriber img_sub;
    ros::Subscriber imu_sub;

    // Publishers.
    image_transport::Publisher vis_img_pub;
    ros::Publisher odom_pub;
    ros::Publisher stable_feature_pub;
    ros::Publisher active_feature_pub;
    ros::Publisher msckf_feature_pub;
    ros::Publisher path_pub;
    
    ros::Publisher poseout_pub;
    bool do_publish_tf_;
    tf2_ros::TransformBroadcaster poseout_tf2_broadcaster_;
    tf2_ros::Buffer tf2_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf2_listener_;

    unsigned int poses_seq_out = 0;

    // Msgs to be published.
    std::queue<std_msgs::Header> header_buffer_;    // buffer for heads of msgs to be published

    // Msgs to be published.
    nav_msgs::Odometry odom_msg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr stable_feature_msg_ptr;
    pcl::PointCloud<pcl::PointXYZ>::Ptr active_feature_msg_ptr;
    pcl::PointCloud<pcl::PointXYZ>::Ptr msckf_feature_msg_ptr;

    nav_msgs::Path path_msg;

    // Frame id
    std::string fixed_frame_id;
    std::string base_frame_id_;
    std::string camera_frame_id_;

    // Pointer for image processer.
    ImageProcessorPtr ImgProcesser;

    // Pointer for estimator.
    OrcVIOPtr Estimator;

    // Directory for files
    std::string config_file;
    std::string result_file;

    // for saving trajectory 
    std::string output_dir_traj;
    std::ofstream fStateToSave;

    // Our groundtruth states
    std::map<double, Eigen::Matrix<double,17,1>> gt_states;

    double summed_rmse_ori;
    double summed_rmse_pos;
    double summed_nees_ori;
    double summed_nees_pos;
    size_t summed_number;
    bool first_pose_flag;
    Eigen::Matrix4d T_from_est_to_gt;

    // flags 
    // whether we load groundtruth or receive a rostopic  
    int load_groundtruth_flag;

    // IMU message buffer.
    std::vector<ImuData> imu_msg_buffer_;
    std::mutex imu_msg_buffer_mutex_;

    // Img message buffer.
    std::queue<ImageDataPtr> img_msg_buffer_;

    /*
        * @brief loadParameters
        *    Load parameters from the parameter server.
        */
    bool loadParameters();

    /*
        * @brief createRosIO
        *    Create ros publisher and subscirbers.
        */
    bool createRosIO();

    /*
        * @brief imageCallback
        *    Callback function for the monocular images.
        * @param image msg.
        */
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    /*
        * @brief imuCallback
        *    Callback function for the imu message.
        * @param msg IMU msg.
        */
    void imuCallback(const sensor_msgs::ImuConstPtr& msg);

    /*
        * @brief function to convert Float64MultiArray to eigen 
        * @param Float64MultiArray msg.
        */
    Eigen::MatrixXd msgToEigen(const std_msgs::Float64MultiArray& msg);

    /*
        * @brief publish Publish the results of VIO.
        * @param time The time stamp of output msgs.
        */
    void publishVIO(const ros::Time& time);

    /*
        * @brief publish Publish the groundtruth
        * @param time The time stamp of output msgs.
        */
    void publishGroundtruth(const ros::Time& time);

    /**
     * @brief Get the root of the tf tree
     * @param frame_id one of the starting frames
     * @param time the time when the tf tree should be sampled
     */
    const std::string find_tf_tree_root(const std::string& frame_id, const ros::Time& time);

};

typedef System::Ptr SystemPtr;
typedef System::ConstPtr SystemConstPtr;

} // end namespace orcvio


#endif  //SYSTEM_H
