#ifndef OBJECTINITNODE_H
#define OBJECTINITNODE_H

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp> 

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <std_msgs/Float64MultiArray.h>
#include <eigen_conversions/eigen_msg.h>

#include <tuple>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include "sort_ros/TrackedBoundingBoxes.h"

#include "orcvio/utils/math_utils.hpp"
#include "orcvio/utils/se3_ops.hpp"
#include "orcvio/feat/FeatureInitializer.h"
#include "orcvio/feat/FeatureInitializerOptions.h"

#include "orcvio/obj/ObjectFeature.h"
#include "orcvio/obj/ObjectFeatureInitializer.h"
#include "orcvio/obj/ObjectState.h"

namespace orcvio {

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class ObjectInitNode
{

  public:

    // we put this in the public 
    // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Destructor.
    ~ObjectInitNode();

    // Initialize the object.
    bool initialize();

    typedef boost::shared_ptr<ObjectInitNode> Ptr;
    typedef boost::shared_ptr<const ObjectInitNode> ConstPtr;

    /**
     * @brief constructor of the class 
     */
    ObjectInitNode(ros::NodeHandle &nh);

    /**
     * @brief initialize the object state initializer 
     * @param object_class object class name 
     * @param object_mean_shape ellipsoid shape prior loaded from yaml 
     * @param object_keypoints_mean semantic keypoints mean shape loaded from yaml
     */
    void setup_object_feature_initializer(const std::string& object_class, const Eigen::Vector3d& object_mean_shape, const Eigen::MatrixX3d& object_keypoints_mean);

    /**
     * @brief callback to receive camera info  
     * @param sensor_msgs/CameraInfo Message  
     */
    void callback_caminfo(const sensor_msgs::CameraInfoConstPtr& cam_info);

    /**
     * @brief callback to handle poses 
     * @param odometry message for poses 
     */
    void callback_pose(const geometry_msgs::PoseStamped::ConstPtr &odom_ptr);

    /**
     * @brief callback to handle semantic measurements  
     * @param message for bounding box and semantic keypoints measurements
     */
    void callback_sem(const sensor_msgs::ImageConstPtr& message,
                const sort_ros::TrackedBoundingBoxesConstPtr& bboxes_msg);

    /**
     * @brief get the timestamps from the state and camera poses 
     *
     * @param state State of the filter
     * @param timestamps for each keypoint 
     * @param clones_cam to hold the camera poses 
     */
    void get_times_cam_poses(const std::vector<double>& clonetimes, std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>>& clones_cam);

    /**
     * @brief do object initialization  
     * @param lost_object_ids a vector of lost object ids
     */
    void do_object_feature_initialization(const std::vector<int>& lost_object_ids);

    /**
     * @brief set track length  
     * @param min track length 
     * @param max track length 
     */
    void set_track_length(const int & min_object_feature_track_length, const int & max_object_feature_track_length);

    /**
     * @brief set groundtruth object state 
     * @param object class 
     * @param object id 
     * @param object position
     * @param object rotation 
     */
    void add_gt_object_state(const std::string& object_class, const int& object_id, const Eigen::MatrixXd & object_position_gt_mat, const Eigen::MatrixXd & object_rotation_gt_mat);

    /**
     * @brief remove object observations that are already optimized 
     * @param lost_object_ids a vector of lost object ids 
     */
    void remove_used_object_obs(std::vector<int>& lost_object_ids);

    /**
     * @brief plot the quadrics  
     */
    void publish_quadrics();

    /**
     * @brief plot the gt objects   
     */
    void publish_gt_objects();

    /**
     * @brief assign color to objects 
     * @param object id 
     * @param color holds the color to be assigned to the object 
     */
    void color_from_id(const int id, std_msgs::ColorRGBA& color);

    /**
     * @brief visualize the objects
     */
    void visualize_map_only();

    /**
     * @brief obtain common timestamps from pose and measurements  
     * @param pose_timestamps_all are all timestamps of poses 
     * @param mea_timestamps_all are all timestamps of semantic measurements 
     * @param common_clonetimes is an empty vector that holds the common timestamps 
     */
    // void obtain_common_timestamps(const std::vector<double>& pose_timestamps_all, const std::vector<double>& mea_timestamps_all, std::vector<double>& common_clonetimes);
    void obtain_common_timestamps(std::vector<double>& pose_timestamps_all, std::vector<double>& mea_timestamps_all, std::vector<double>& common_clonetimes);

    /**
     * @brief save the estimated keypoints to file  
     * @param object id 
     * @param estimated keypoints  
     * @param file path 
     */
    void save_kps_to_file(const int & object_id, const Eigen::Matrix3Xd & valid_shape_global_frame, std::string filepath_format);

    /**
     * @brief publish the kp track image   
     * @param image to publish 
     */
    void publish_track_image(const cv::Mat& track_image);

    /**
     * @brief convert the quadrics to sphere for the lite version 
     * @param quadrics to convert 
     */
    void convert_quad_to_sphere(std::vector<double>& mean_shape);

    /**
     * @brief draw bounding box on the image 
     * @param image to draw 
     */
    void draw_bbox(cv::Mat& img, double xmin, double ymin, double xmax, double ymax);

  private:

    // publishers 
    ros::Publisher pub_quadrics;
    ros::Publisher pub_gt_objects;

    // for plotting 
    std::string track_image_topic;
    std::unique_ptr<image_transport::ImageTransport> image_trans;
    image_transport::Publisher trackImagePublisher;

    // subscribers  
    std::string topic_bbox;
    std::string topic_pose;
    std::string topic_caminfo;
    // for plotting 
    std::string topic_image;

    ros::Subscriber sub_gtpose;
    // ros::Subscriber sub_sem;
    ros::Subscriber sub_caminfo;
    // for plotting 
    std::unique_ptr<message_filters::Subscriber<sort_ros::TrackedBoundingBoxes>> sub_sem;
    // std::unique_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, sort_ros::TrackedBoundingBoxes>> sub_sem_img;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sort_ros::TrackedBoundingBoxes> MySyncPolicy;
    std::unique_ptr<message_filters::Synchronizer<MySyncPolicy> > sub_sem_img;

    std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_img;

    Eigen::Matrix4d T_CtoI;
    Eigen::Matrix4d T_ItoC;

    cv::Matx33d camK;
    cv::Vec4d camD;

    bool first_pose_flag;
    bool set_first_pose_as_origin_flag;

    Eigen::Vector3d first_uav_translation_gt;
    Eigen::Matrix3d R0;
    Eigen::Vector3d p0;
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_imu;
    std::vector<double> pose_timestamps;

    std::unordered_map<int, std::shared_ptr<ObjectFeature>> object_obs_dict;

    /// Many to one mapping to standardize class names For example, truck ->
    /// car, van -> car, bench -> chair, chair -> chair, car -> car.
    std::map<std::string, std::string >  object_standardized_class_name_;

    /// Standardized names -> ObjectFeatureInitializer
    std::map<std::string, std::shared_ptr<ObjectFeatureInitializer>> all_objects_feat_init_;

    /// Standardized names -> Marker color as RGB vector for visualization
    std::map<std::string, std_msgs::ColorRGBA> object_marker_colors_;

    unsigned max_object_feature_track_length;
    unsigned min_object_feature_track_length;
    std::unordered_map<int, ObjectState> all_object_states_dict;

    // gt object states
    bool load_gt_object_info_flag;
    std::vector<int> object_id_gt_vec;
    std::vector<std::string> object_class_gt_vec;
    orcvio::vector_eigen<Eigen::MatrixXd> object_position_gt_vec;
    orcvio::vector_eigen<Eigen::MatrixXd> object_rotation_gt_vec;

    // object gt sizes for unity 
    std::unordered_map<std::string, std::vector<double>> object_sizes_gt_dict;

    // file path 
    // for saving object map 
    std::string result_dir_path_object_map;

    /// If we should use Levenberg Marquardt for object pose fine turing
    bool do_fine_tune_object_pose_using_lm = false; 

    // whether we use the new bounding box residual 
    bool use_new_bbox_residual_flag = false; 

    // whether we use left or right perturbation  
    bool use_left_perturbation_flag = true; 

    FeatureInitializerOptions featinit_options;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // flag to check if we are using unity 
    // in unity there is no front end, as the observations 
    // directly come from groundtruth 
    bool use_unity_dataset_flag;

    // flag to check whether we are using bbox only, 
    // without using keypoints, if so, then it's the lite version
    bool use_bbox_only_flag; 

}; // ObjectInitNode class 

typedef ObjectInitNode::Ptr ObjectInitPtr;
typedef ObjectInitNode::ConstPtr ObjectInitConstPtr;

} // namespace orcvio

#endif //OBJECTINITNODE_H
