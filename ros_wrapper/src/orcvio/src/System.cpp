//
// Managing the image processer and the estimator.
//

#include <sensor_msgs/PointCloud2.h>

#include <System.h>

#include <iostream>

#include <boost/filesystem.hpp>
#include <rosbag/bag.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
namespace bfs = boost::filesystem;

namespace tf2 {
    void convert(const geometry_msgs::Point& position, geometry_msgs::Vector3& translation)
    {
        translation.x = position.x;
        translation.y = position.y;
        translation.z = position.z;
    }

    void convert(const geometry_msgs::Pose& pose, geometry_msgs::Transform& transform)
    {
        transform.rotation = pose.orientation;
        convert(pose.position, transform.translation);
    }

}

namespace orcvio {


System::System(ros::NodeHandle& n) : nh(n) {
}


System::~System() {
    // close file 
    fStateToSave.close();
}


// Load parameters from launch file
bool System::loadParameters() {

    ROS_INFO("System: Start loading ROS parameters...");

    nh.getParam("result_file", result_file);
    nh.param<std::string>("output_dir_traj", output_dir_traj, "./cache");

    // config file to be used for orcvio 
    nh.getParam("config_file", config_file);
    if (! bfs::exists(config_file)) {
        ROS_FATAL_STREAM("Unable to locate " << config_file);
        throw std::runtime_error("Unable to locate: " + config_file);
    }

    summed_rmse_ori = 0.0;
    summed_rmse_pos = 0.0;
    summed_nees_ori = 0.0;
    summed_nees_pos = 0.0;
    summed_number = 0;

    // for publishing groundtruth
    first_pose_flag = false; 

    return true;
}


// Subscribe image and imu msgs.
bool System::createRosIO() {
    // Subscribe imu msg.
    imu_sub = nh.subscribe("imu", 5000, &System::imuCallback, this);

    // Subscribe image msg.
    img_sub = nh.subscribe("cam0_image", 50, &System::imageCallback, this);

    // Advertise processed image msg.
    image_transport::ImageTransport it(nh);
    vis_img_pub = it.advertise("visualization_image", 1);

    // Advertise odometry msg.
    odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);

    poseout_pub = nh.advertise<geometry_msgs::PoseStamped>("poseout", 2);
    ROS_INFO("Publishing: %s", poseout_pub.getTopic().c_str());

    // Advertise point cloud msg.
    stable_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
            "stable_feature_point_cloud", 1);
    active_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
            "active_feature_point_cloud", 1);
    msckf_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
            "msckf_feature_point_cloud", 1);

    // Advertise path msg.
    path_pub = nh.advertise<nav_msgs::Path>("path", 10);

    nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
    nh.param<string>("base_frame_id", base_frame_id_, "");

    nh.param<bool>("publish_tf", do_publish_tf_, true);

    stable_feature_msg_ptr.reset(
        new pcl::PointCloud<pcl::PointXYZ>());
    stable_feature_msg_ptr->header.frame_id = fixed_frame_id;
    stable_feature_msg_ptr->height = 1;

    fStateToSave.open((output_dir_traj+"/stamped_traj_estimate.txt").c_str(), std::ofstream::trunc);

    tf2_listener_.reset(new tf2_ros::TransformListener(tf2_buffer_));

    return true;
}


// Initializing the system.
bool System::initialize() {
    // Load necessary parameters
    if (!loadParameters())
        return false;
    ROS_INFO("System: Finish loading ROS parameters...");

    if (load_groundtruth_flag)
    {

        // load groundtruth
        std::string path_to_gt;
        nh.param<std::string>("path_gt", path_to_gt, "");
        DatasetReader::load_gt_file(path_to_gt, gt_states);

    }

    // Set pointers of image processer and estimator.
    ImgProcesser.reset(new ImageProcessor(config_file));
    Estimator.reset(new OrcVIO(config_file));

    // Initialize image processer and estimator.
    if (!ImgProcesser->initialize()) {
        ROS_WARN("Image Processer initialization failed!");
        return false;
    }
    if (!Estimator->initialize()) {
        ROS_WARN("Estimator initialization failed!");
        return false;
    }

    // Try subscribing msgs
    if (!createRosIO())
        return false;
    ROS_INFO("System Manager: Finish creating ROS IO...");

    return true;
}


// Push imu msg into the buffer.
void System::imuCallback(const sensor_msgs::ImuConstPtr& msg) {

    // for debugging 
    // std::cout << "imu cb timestamp: " << msg->header.stamp.toSec() << std::endl; 
    const std::lock_guard<std::mutex> lock(imu_msg_buffer_mutex_);

    imu_msg_buffer_.push_back(
        ImuData(
            msg->header.stamp.toSec(),
            msg->angular_velocity.x,
            msg->angular_velocity.y,
            msg->angular_velocity.z,
            msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z));
}


// Process the image and trigger the estimator.
void System::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    // test
    cv_bridge::CvImageConstPtr cvCPtr = cv_bridge::toCvShare(
        msg,
        sensor_msgs::image_encodings::MONO8);
    orcvio::ImageDataPtr msgPtr(new ImgData);
    msgPtr->timeStampToSec = cvCPtr->header.stamp.toSec();
    msgPtr->image = cvCPtr->image.clone();
    std_msgs::Header header = cvCPtr->header;
    camera_frame_id_ = cvCPtr->header.frame_id;

    img_msg_buffer_.push(msgPtr);
    header_buffer_.push(header);

    // Do nothing if no imu msg is received.
    if (imu_msg_buffer_.empty()) {
        return;
    }

    while ( ! img_msg_buffer_.empty()) {
        orcvio::ImageDataPtr imgPtr = img_msg_buffer_.front();
        img_msg_buffer_.pop();
        auto header = header_buffer_.front();
        header_buffer_.pop();

        MonoCameraMeasurementPtr features(new MonoCameraMeasurement());

        // Process image to get feature measurement.
        // return false if no feature 
        bool bProcess = ImgProcesser->processImage(msgPtr, imu_msg_buffer_, features);

        // Filtering if get processed feature.
        // return false if feature update not begin 
        bool bPubOdo = false;

        if (bProcess) {
            const std::lock_guard<std::mutex> lock(imu_msg_buffer_mutex_);
            bPubOdo = Estimator->processFeatures(features, imu_msg_buffer_);
        }

        // Publish msgs if necessary
        if (bProcess) {
            cv_bridge::CvImage _image(header, "bgr8", ImgProcesser->getVisualImg());
            vis_img_pub.publish(_image.toImageMsg());
        }

        if (bPubOdo) {
            publishVIO(header.stamp);

            if (load_groundtruth_flag)
                publishGroundtruth(header.stamp);

        }
    }
}

Eigen::MatrixXd System::msgToEigen(const std_msgs::Float64MultiArray& msg)
{
	double dstride0 = msg.layout.dim[0].stride;
	double dstride1 = msg.layout.dim[1].stride;
	double h = msg.layout.dim[0].size;
	double w = msg.layout.dim[1].size;

	// Below are a few basic Eigen demos:
	std::vector<double> data = msg.data;
	Eigen::Map<Eigen::MatrixXd> mat(data.data(), h, w);
	// std::cout << "I received = " << std::endl << mat << std::endl;
	
	return mat;
}

const std::string System::find_tf_tree_root(
    const std::string& frame_id, const ros::Time& time)
{
    std::string cursor = frame_id;
    std::string parent;
    while (tf2_buffer_._getParent(cursor, time, parent))
        cursor = parent;
    // ROS_WARN_STREAM("Found root : " << cursor);
    return cursor;
}

// Publish informations of VIO, including odometry, path, points cloud and whatever needed.
void System::publishVIO(const ros::Time& time) {

    // construct odometry msg

    odom_msg.header.stamp = time;
    odom_msg.header.frame_id = fixed_frame_id;
    odom_msg.child_frame_id = camera_frame_id_;
    Eigen::Isometry3d T_b_w = Estimator->getTbw();
    Eigen::Vector3d body_velocity = Estimator->getVel();
    Matrix<double, 6, 6> P_body_pose = Estimator->getPpose();
    Matrix3d P_body_vel = Estimator->getPvel();

    // use IMU pose 
    // tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
    // use camera pose 
    Eigen::Isometry3d T_c_w = Estimator->getTcw();
    tf::poseEigenToMsg(T_c_w, odom_msg.pose.pose);

    tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

    // construct path msg
    path_msg.header.stamp = time;
    path_msg.header.frame_id = fixed_frame_id;
    geometry_msgs::PoseStamped curr_path;
    curr_path.header.stamp = time;
    curr_path.header.frame_id = fixed_frame_id;
    tf::poseEigenToMsg(T_b_w, curr_path.pose);
    path_msg.poses.push_back(curr_path);

    // construct pose msg
    // from imu to global 
    // this is used by object mapping 
    geometry_msgs::PoseStamped wTi_msg;
    wTi_msg.header.stamp = time;

    wTi_msg.header.frame_id = camera_frame_id_;
    // wTi_msg.header.frame_id = fixed_frame_id;
    
    wTi_msg.header.seq = poses_seq_out;
    // use IMU pose 
    tf::poseEigenToMsg(T_b_w, wTi_msg.pose);

    // construct point cloud msg
    // Publish the 3D positions of the features.
    // Including stable and active ones.
    // --Stable features
    std::map<orcvio::FeatureIDType,Eigen::Vector3d> StableMapPoints;
    Estimator->getStableMapPointPositions(StableMapPoints);
    for (const auto& item : StableMapPoints) {
        const auto& feature_position = item.second;
        stable_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
    }
    stable_feature_msg_ptr->width = stable_feature_msg_ptr->points.size();
    // --Active features
    active_feature_msg_ptr.reset(
        new pcl::PointCloud<pcl::PointXYZ>());
    active_feature_msg_ptr->header.frame_id = fixed_frame_id;
    active_feature_msg_ptr->height = 1;
    std::map<orcvio::FeatureIDType, Eigen::Vector3d> ActiveMapPoints;
    Estimator->getActiveMapPointPositions(ActiveMapPoints);
    for (const auto& item : ActiveMapPoints) {
        const auto& feature_position = item.second;
        active_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
    }
    active_feature_msg_ptr->width = active_feature_msg_ptr->points.size();

    // publish msckf features
    msckf_feature_msg_ptr.reset(
        new pcl::PointCloud<pcl::PointXYZ>());
    msckf_feature_msg_ptr->header.frame_id = fixed_frame_id;
    msckf_feature_msg_ptr->height = 1;
    std::map<orcvio::FeatureIDType, Eigen::Vector3d> MSCKFPoints;
    Estimator->getMSCKFMapPointPositions(MSCKFPoints);
    for (const auto& item : MSCKFPoints) {
        const auto& feature_position = item.second;
        msckf_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
    }
    msckf_feature_msg_ptr->width = msckf_feature_msg_ptr->points.size();

    odom_pub.publish(odom_msg);
    path_pub.publish(path_msg);
    poseout_pub.publish(wTi_msg);
    if (do_publish_tf_) {
        // Subscript notation for transforms, (i)mu, (c)amera, robot-(b)ase, (w)orld
        // wTi is transform that converts from i -> w: p_w = wTi @ p_i
        Eigen::Isometry3d wTc_eigen = T_c_w;
        Eigen::Isometry3d cTb_eigen;
        try{
            if (! base_frame_id_.size())
                base_frame_id_ = find_tf_tree_root(camera_frame_id_, time);
            geometry_msgs::TransformStamped cTb_msg =
                tf2_buffer_.lookupTransform(camera_frame_id_, base_frame_id_,
                                            time,
                                            /*timeout=*/ros::Duration(2));
            cTb_eigen = tf2::transformToEigen(cTb_msg.transform);
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN_STREAM("Unable to get " << base_frame_id_
                            << " -> " <<  camera_frame_id_ << " transform." <<
                            ex.what() << ". Assuming identity. ");
            // Not found
            // Set to identity transform
            cTb_eigen = Eigen::Isometry3d::Identity();
        }
        Eigen::Isometry3d wTb_eigen = wTc_eigen * cTb_eigen;
        geometry_msgs::TransformStamped transform = tf2::eigenToTransform(wTb_eigen);
        transform.header.seq = poses_seq_out;
        transform.header.stamp = time;
        transform.header.frame_id = fixed_frame_id; // (w)orld
        transform.child_frame_id = base_frame_id_; // robot-(b)ase
        // http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20broadcaster%20%28C%2B%2B%29
        // If the position of child origin in world frame acts as translation, then
        // the transform transforms from child_frame_id -> header.frame_id
        // unlike what is documented in geometry_msgs/TransformStamped.
        // http://docs.ros.org/en/api/geometry_msgs/html/msg/TransformStamped.html
        poseout_tf2_broadcaster_.sendTransform(transform);
    }
    // Move them forward in time
    poses_seq_out++;

    stable_feature_pub.publish(stable_feature_msg_ptr);
    active_feature_pub.publish(active_feature_msg_ptr);
    msckf_feature_pub.publish(msckf_feature_msg_ptr);

    // save the pose to txt for trajectory evaluation 
    // timestamp tx ty tz qx qy qz qw
    fStateToSave << std::fixed << std::setprecision(3) << curr_path.header.stamp.toSec() << " "
        << curr_path.pose.position.x << " " << curr_path.pose.position.y << " " << curr_path.pose.position.z << " "
        << curr_path.pose.orientation.x << " " << curr_path.pose.orientation.y << " " << curr_path.pose.orientation.z << " " << curr_path.pose.orientation.w << std::endl; 

}

void System::publishGroundtruth(const ros::Time& time) {

    double timestamp = time.toSec();
    // Our groundtruth state
    Eigen::Matrix<double,17,1> state_gt;

    // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
    if(!DatasetReader::get_gt_state(timestamp, state_gt, gt_states)) {
        return;
    }

    Eigen::Vector4d q_gt;
    Eigen::Vector3d p_gt;

    q_gt << state_gt(1,0),state_gt(2,0),state_gt(3,0),state_gt(4,0);
    p_gt << state_gt(5,0), state_gt(6,0), state_gt(7,0);

    Eigen::Isometry3d T_b_w = Estimator->getTbw();

    if (!first_pose_flag)
    {
        // load the first pose 
        Eigen::Matrix4d first_pose_gt;
        first_pose_gt.block<3, 3>(0, 0) = quaternionToRotation(q_gt);
        first_pose_gt.block<3, 1>(0, 3) = p_gt;

        T_from_est_to_gt = first_pose_gt * T_b_w.inverse().matrix();

        first_pose_flag = true;
    }

    Eigen::Matrix4d T_est_corrected = T_from_est_to_gt * T_b_w.matrix();
    Eigen::Matrix3d wRi = T_est_corrected.block<3, 3>(0, 0);
    Eigen::Vector3d wPi = T_est_corrected.block<3, 1>(0, 3);

    // Difference between positions
    double dx = wPi(0)-p_gt(0);
    double dy = wPi(1)-p_gt(1);
    double dz = wPi(2)-p_gt(2);
    double rmse_pos = std::sqrt(dx*dx+dy*dy+dz*dz);

    // Quaternion error
    Eigen::Matrix<double,4,1> quat_st, quat_diff;

    quat_st = rotationToQuaternion(wRi);   
    Eigen::Vector4d quat_gt_inv = inverseQuaternion(q_gt);
    quat_diff = quaternionMultiplication(quat_st, quat_gt_inv);
    double rmse_ori = (180/M_PI)*2*quat_diff.block(0,0,3,1).norm();

    // Update our average variables
    summed_rmse_ori += rmse_ori;
    summed_rmse_pos += rmse_pos;
    summed_number++;

    FILE* fp = fopen(result_file.c_str(), "w");
    fprintf(fp, "%f %f\n", summed_rmse_ori/summed_number, summed_rmse_pos/summed_number);
    fclose(fp);

}

} // end namespace orcvio
