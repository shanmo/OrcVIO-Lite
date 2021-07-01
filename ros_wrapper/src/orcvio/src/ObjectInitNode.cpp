#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <set>

#include <ObjectInitNode.h>

using Eigen::Matrix3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using namespace boost::filesystem;

namespace orcvio
{

    ObjectInitNode::~ObjectInitNode()
    {

    }

    ObjectInitNode::ObjectInitNode(ros::NodeHandle &nh)
    {

        // quadrics publishing
        pub_quadrics = nh.advertise<visualization_msgs::MarkerArray>("/orcvio/quadrics", 2);
        ROS_INFO("Publishing: %s", pub_quadrics.getTopic().c_str());

        pub_gt_objects = nh.advertise<visualization_msgs::MarkerArray>("/orcvio/gt_objects", 2);
        ROS_INFO("Publishing: %s", pub_gt_objects.getTopic().c_str());

        // track image publishing
        nh.param<std::string>("track_image_topic", track_image_topic, "/orcvio/track_image");

        nh.param<std::string>("fixed_frame_id", fixed_frame_id, "world");
        // nh.param<std::string>("child_frame_id", child_frame_id, "world");

        // Create subscribers
        nh.param<std::string>("topic_bbox", topic_bbox, "/yolo/bbox");
        nh.param<std::string>("topic_image", topic_image, "/husky/camera/image");
        nh.param<std::string>("topic_pose", topic_pose, "/unity_ros/husky/TrueState/odom");
        nh.param<std::string>("topic_caminfo", topic_caminfo, "/husky/camera/camera_info");

        // get the dir path to save object map
        std::string ros_log_dir;
        if ( ! ros::get_environment_variable(ros_log_dir, "ROS_LOG_DIR") ) {
            if ( ros::get_environment_variable(ros_log_dir, "HOME") ) {
                ros_log_dir = ros_log_dir + "/.ros";
            } else {
                ROS_FATAL("Environment variable HOME is not set");
            }
        }
        nh.param<std::string>("result_dir_path_object_map", result_dir_path_object_map, ros_log_dir);

        if (exists(result_dir_path_object_map))
        {
            directory_iterator end_it;
            directory_iterator it(result_dir_path_object_map.c_str());
            if (it == end_it)
            {
                // this is fine 
            }
            else 
            {
                ROS_INFO_STREAM("object map path exists and nonempty, delete contents in " << result_dir_path_object_map.c_str());
                // if this dir already exists, then delete all contents inside
                std::string del_cmd = "exec rm -r " + result_dir_path_object_map + "*";
                int tmp = system(del_cmd.c_str());
            }
        }
        else
        {
            // if this dir does not exist, create the dir
            const char *path = result_dir_path_object_map.c_str();
            boost::filesystem::path dir(path);
            if (boost::filesystem::create_directories(dir))
            {
                ROS_INFO_STREAM("Directory Created: " << result_dir_path_object_map.c_str());
            } else {
                ROS_FATAL("Unable to create directory ");
            }
        }

        sub_caminfo = nh.subscribe(topic_caminfo.c_str(), 9999, &ObjectInitNode::callback_caminfo, this);
        sub_gtpose = nh.subscribe(topic_pose.c_str(), 9999, &ObjectInitNode::callback_pose, this);

        // for plotting
        sub_sem = make_unique<message_filters::Subscriber<sort_ros::TrackedBoundingBoxes> >(nh, topic_bbox, 1);
        sub_img = make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh, topic_image, 1);
        namespace sph = std::placeholders; // for _1, _2, ...
        // sub_sem_img = make_unique<message_filters::TimeSynchronizer<sensor_msgs::Image, sort_ros::TrackedBoundingBoxes> >(*sub_img, *sub_sem, 10);
        sub_sem_img = make_unique<message_filters::Synchronizer<MySyncPolicy> > (
            MySyncPolicy(100), *sub_img, *sub_sem);
        ROS_INFO_STREAM("Subscribing callback_sem to " << topic_image << " and " << topic_bbox);
        sub_sem_img->registerCallback(std::bind(&ObjectInitNode::callback_sem, this, sph::_1, sph::_2));

        image_trans = make_unique<image_transport::ImageTransport>(nh);
        trackImagePublisher = image_trans->advertise(track_image_topic, 10);

        // Our camera extrinsics transform used in getting camera pose
        // since VIO only outputs IMU pose but object LM needs camera pose
        // Read in from ROS, and save into our eigen mat
        std::vector<double> matrix_TItoC;
        std::vector<double> matrix_TItoC_default = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        int i = 0;
        nh.param<std::vector<double>>("T_cam_imu", matrix_TItoC, matrix_TItoC_default);
        // step 1: load T_ItoC, this means from imu to cam
        T_ItoC << matrix_TItoC.at(0), matrix_TItoC.at(1), matrix_TItoC.at(2), matrix_TItoC.at(3),
            matrix_TItoC.at(4), matrix_TItoC.at(5), matrix_TItoC.at(6), matrix_TItoC.at(7),
            matrix_TItoC.at(8), matrix_TItoC.at(9), matrix_TItoC.at(10), matrix_TItoC.at(11),
            matrix_TItoC.at(12), matrix_TItoC.at(13), matrix_TItoC.at(14), matrix_TItoC.at(15);

        // step 2: get T_CtoI
        T_CtoI = T_ItoC.inverse().eval();


        // #--------------------------------------------------------------------------------------------
        // # different object classes start 
        // #--------------------------------------------------------------------------------------------

        XmlRpc::XmlRpcValue object_classes;
        const std::string OBJECT_CLASSES_KEY = "object_classes";
        nh.getParam(OBJECT_CLASSES_KEY, object_classes);
        ROS_ASSERT(object_classes.getType() == XmlRpc::XmlRpcValue::TypeStruct);
        for (XmlRpc::XmlRpcValue::const_iterator key_value =  object_classes.begin();
             key_value != object_classes.end();
             ++key_value)
            {
            auto const& class_name = static_cast<const std::string>(key_value->first);
            ROS_INFO_STREAM("Adding object class: " << class_name);

            std::vector<double> object_mean_shape_vec(3);
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/object_mean_shape", object_mean_shape_vec);
            
            // change quadric to sphere for initialization 
            convert_quad_to_sphere(object_mean_shape_vec);
            
            object_sizes_gt_dict[class_name] = object_mean_shape_vec;
            int kps_num;
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/keypoints_num", kps_num);
            std::vector<double> object_keypoints_mean_vec(3 * kps_num);
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/object_keypoints_mean", object_keypoints_mean_vec);
            Eigen::MatrixXd object_mean_shape_mat = Eigen::Map<Eigen::Matrix<double, 3, 1>>(object_mean_shape_vec.data());
            Eigen::MatrixXd object_keypoints_mean_mat = Eigen::Map<Eigen::MatrixXd>(
                object_keypoints_mean_vec.data(), kps_num, 3);
            std::vector<std::string> object_accepted_names;
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/aliases", object_accepted_names);
            object_accepted_names.push_back(class_name);
            for (auto const& name: object_accepted_names)
                object_standardized_class_name_[name] = class_name;
            setup_object_feature_initializer(class_name,
                                             object_mean_shape_mat, object_keypoints_mean_mat);

            std::vector<double> marker_color;
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/marker_color", marker_color);
            std_msgs::ColorRGBA marker_color_rgba;
            marker_color_rgba.r = marker_color[0];
            marker_color_rgba.g = marker_color[1];
            marker_color_rgba.b = marker_color[2];
            marker_color_rgba.a = 1.0;
            object_marker_colors_[class_name] = marker_color_rgba;
        }

        int max_object_feature_track_length;
        int min_object_feature_track_length;
        nh.getParam("max_object_feature_track", max_object_feature_track_length);
        nh.getParam("min_object_feature_track", min_object_feature_track_length);
        set_track_length(min_object_feature_track_length, max_object_feature_track_length);

        // determine which dataset we are using
        nh.param<bool>("use_unity_dataset_flag", use_unity_dataset_flag, false);

        // load the flag for object LM
        nh.param<bool>("do_fine_tune_object_pose_using_lm", do_fine_tune_object_pose_using_lm, false);

        // load the flag for new bounding box residual
        nh.param<bool>("use_new_bbox_residual_flag", use_new_bbox_residual_flag, false);

        // load the flag for using left or right perturbation
        nh.param<bool>("use_left_perturbation_flag", use_left_perturbation_flag, true);

        // set_first_pose_as_origin_flag = true;
        set_first_pose_as_origin_flag = false;

        first_pose_flag = true;

        // load gt object states
        nh.param<bool>("load_gt_object_info_flag", load_gt_object_info_flag, true);

        if (load_gt_object_info_flag)
        {

            int total_object_num = 0;
            nh.getParam("total_object_num", total_object_num);

            std::vector<int> objects_ids_gt(total_object_num);
            nh.getParam("objects_ids_gt", objects_ids_gt);

            std::vector<std::string> objects_class_gt(total_object_num);
            nh.getParam("objects_class_gt", objects_class_gt);

            std::vector<double> objects_rotation_gt(9 * total_object_num);
            nh.getParam("objects_rotation_gt", objects_rotation_gt);

            std::vector<double> objects_translation_gt(3 * total_object_num);
            nh.getParam("objects_translation_gt", objects_translation_gt);

            std::vector<double> object_position_gt(3);
            std::vector<double> object_rotation_gt(9);

            for (int i = 0; i < total_object_num; i++)
            {

                // All sequence containers in C++ preserve internal order
                int object_id_gt = objects_ids_gt[i];
                std::string object_class_gt = objects_class_gt[i];

                object_position_gt = std::vector<double>(objects_translation_gt.begin() + (i * 3), objects_translation_gt.begin() + ((i + 1) * 3));
                object_rotation_gt = std::vector<double>(objects_rotation_gt.begin() + (i * 9), objects_rotation_gt.begin() + ((i + 1) * 9));

                Eigen::MatrixXd object_position_gt_mat = Eigen::Map<Eigen::Matrix<double, 3, 1>>(object_position_gt.data());
                Eigen::MatrixXd object_rotation_gt_mat = Eigen::Map<Eigen::Matrix<double, 3, 3>>(object_rotation_gt.data());

                // for debugging
                // for (const auto & i : object_rotation_gt)
                //     std::cout << "object_rotation_gt " << i << std::endl;
                // std::cout << "object_rotation_gt_mat " << object_rotation_gt_mat << std::endl;

                add_gt_object_state(object_class_gt, object_id_gt, object_position_gt_mat, object_rotation_gt_mat);
            }

            std::vector<double> first_uav_translation_gt_vec(3);
            nh.getParam("first_uav_translation_gt", first_uav_translation_gt_vec);
            first_uav_translation_gt = Eigen::Map<Eigen::Matrix<double, 3, 1>>(first_uav_translation_gt_vec.data());
        }

    } // end of ObjectInitNode

    void ObjectInitNode::add_gt_object_state(const std::string &object_class, const int &object_id, const Eigen::MatrixXd &object_position_gt_mat, const Eigen::MatrixXd &object_rotation_gt_mat)
    {
        object_id_gt_vec.push_back(object_id);
        object_class_gt_vec.push_back(object_class);

        object_position_gt_vec.push_back(object_position_gt_mat);
        object_rotation_gt_vec.push_back(object_rotation_gt_mat);
    }

    void ObjectInitNode::set_track_length(const int &min_object_feature_track_length, const int &max_object_feature_track_length)
    {

        this->max_object_feature_track_length = (unsigned)max_object_feature_track_length;
        this->min_object_feature_track_length = (unsigned)min_object_feature_track_length;
    }

    void ObjectInitNode::setup_object_feature_initializer(const std::string &object_class, const Eigen::Vector3d &object_mean_shape, const Eigen::MatrixX3d &object_keypoints_mean)
    {
        Eigen::Matrix<double, 3, 3> camera_intrinsics;
        cv2eigen(camK, camera_intrinsics);

        auto class_name_iter = object_standardized_class_name_.find(object_class);
        if (class_name_iter == object_standardized_class_name_.end()) {
            // unknown object class

            ROS_WARN_STREAM_ONCE_NAMED(
                "UnknownClass" + object_class,
                "Ignoring unknown object class: " <<
                object_class);
            return;
        }
        auto obj_feat_init_iter = all_objects_feat_init_.find(class_name_iter->second);
        if (obj_feat_init_iter == all_objects_feat_init_.end()) { 
            all_objects_feat_init_[class_name_iter->second] =
                std::make_shared<ObjectFeatureInitializer>(
                    featinit_options, object_mean_shape, object_keypoints_mean,
                    camera_intrinsics);
        } else {
            obj_feat_init_iter->second.reset(
                new ObjectFeatureInitializer(
                    featinit_options, object_mean_shape, object_keypoints_mean,
                    camera_intrinsics));
        }
    }

    void ObjectInitNode::convert_quad_to_sphere(std::vector<double> &mean_shape)
    {
        // if the mean shape has equal width and length, then do not do conversion 
        // if the mean shape has different width and length, then set those to the average 
        // do not change height during conversion 

        if (mean_shape[0] == mean_shape[1])
            return; 
        else 
        {
            double avg = (mean_shape[0] + mean_shape[1])/2;
            mean_shape[0] = avg;
            mean_shape[1] = avg; 
        }

    }

    // this is for general case
    void ObjectInitNode::callback_caminfo(const sensor_msgs::CameraInfoConstPtr &cam_info)
    {
        // for intrinsics
        std::vector<double> cam0_intrinsics_temp(4);
        std::vector<double> cam0_distortion_coeffs_temp(4);

        camK << cam_info->K[0], 0, cam_info->K[2],
            0, cam_info->K[4], cam_info->K[5],
            0, 0, 1;

        camD << cam_info->D[0], cam_info->D[1], cam_info->D[2], cam_info->D[3];

        // for debugging
        // std::cout << camK << std::endl;
        // std::cout << camD << std::endl;

        sub_caminfo.shutdown();
    }

    void ObjectInitNode::callback_pose(const geometry_msgs::PoseStamped::ConstPtr &odom_ptr)
    {

        if (first_pose_flag)
        {
            p0(0, 0) = odom_ptr->pose.position.x;
            p0(1, 0) = odom_ptr->pose.position.y;
            p0(2, 0) = odom_ptr->pose.position.z;

            // convert to quaternion
            Eigen::Quaterniond q0;
            q0.x() = odom_ptr->pose.orientation.x;
            q0.y() = odom_ptr->pose.orientation.y;
            q0.z() = odom_ptr->pose.orientation.z;
            q0.w() = odom_ptr->pose.orientation.w;

            // convert to rotation matrix
            R0 = q0.normalized().toRotationMatrix();

            first_pose_flag = false;
        }

        // convert to quaternion
        Eigen::Quaterniond q1;
        q1.x() = odom_ptr->pose.orientation.x;
        q1.y() = odom_ptr->pose.orientation.y;
        q1.z() = odom_ptr->pose.orientation.z;
        q1.w() = odom_ptr->pose.orientation.w;

        // convert to rotation matrix
        Matrix3d R1;
        R1 = q1.normalized().toRotationMatrix();

        // normalize rotation
        Matrix3d R1_normalized;
        if (set_first_pose_as_origin_flag)
            // set the first pose as origin
            R1_normalized = R0.transpose() * R1;
        else
            // do NOT set the first pose as origin
            R1_normalized = R1;

        // set first pose as origin
        Eigen::Vector3d t1, t1_new;
        t1 << odom_ptr->pose.position.x, odom_ptr->pose.position.y, odom_ptr->pose.position.z;

        if (set_first_pose_as_origin_flag)
            // set the first pose as origin
            t1_new = R0.transpose() * (t1 - p0);
        else
            // do NOT set the first pose as origin
            t1_new = t1;

        // std::cout << "q1 " << q1.x() << " " << q1.y() << " " << q1.z() << " " << q1.w() << std::endl;
        // std::cout << "t1_new " << t1_new << std::endl;

        // Get current camera pose
        Eigen::Matrix<double, 3, 3> R_ItoG;
        R_ItoG = R1_normalized;

        // std::cout << "insert R " << R_ItoG << std::endl;

        Eigen::Matrix<double, 3, 1> p_IinG;
        p_IinG(0, 0) = t1_new(0, 0);
        p_IinG(1, 0) = t1_new(1, 0);
        p_IinG(2, 0) = t1_new(2, 0);

        // Append to our map
        clones_imu.insert({odom_ptr->header.stamp.toSec(), FeatureInitializer::ClonePose(R_ItoG, p_IinG)});
        pose_timestamps.push_back(odom_ptr->header.stamp.toSec());
    }

    /**
    * @brief print the lost flag  
    * @param boolean lost flag 
    */
    inline const std::string BoolToString(bool b) { return b ? "true" : "false"; }

    // must use a ConstPtr callback to use zero-copy transport
    void ObjectInitNode::callback_sem(const sensor_msgs::ImageConstPtr &message,
                                      const sort_ros::TrackedBoundingBoxesConstPtr &bboxes_msg)
    {

        if (cv::countNonZero(camK) == 0)
        {
            ROS_WARN_STREAM("camK is still 0. Dropping this stream of bounding box messages");
            return;
        }

        // display tracking results
        child_frame_id = message->header.frame_id;
        cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(message, "bgr8");
        cv::Mat track_image = img->image;

        // we only use 1 camera for semantic observations
        int cam_id = 0;
        std::vector<int> lost_object_ids;

        for (const auto &bbox : bboxes_msg->bounding_boxes)
        {

            // check whether the object has already been optimized 
            // if we encounter the same object again, do not track and 
            // do not do optimization for the second time 
            // for now this is mainly used in the Unity dataset 
            if (all_object_states_dict.find(bbox.id) != all_object_states_dict.end())
            {
                // object exists in the current map, skip tracking and optimization 
                continue; 
            }

            // check whether the object class is valid 
            std::string standard_object_class; 
            auto class_name_iter = object_standardized_class_name_.find(bbox.Class);
            if (class_name_iter == object_standardized_class_name_.end()) {
                // unknown object class
                ROS_WARN_STREAM_ONCE_NAMED(
                    "UnknownClass" + bbox.Class,
                    "Ignoring unknown object class: " <<
                    bbox.Class);
                continue;
            }
            else 
            {
                standard_object_class = class_name_iter->second;
            }

            std::shared_ptr<ObjectFeature> obj_obs_ptr;

            // for debugging
            // ROS_INFO_STREAM("bbox id " << bbox.id);

            // check if the object appears the first time
            if (object_obs_dict.find(bbox.id) == object_obs_dict.end())
            {
                obj_obs_ptr.reset(new ObjectFeature(bbox.id, standard_object_class));

                // for debugging
                // std::cout << "New object id " << obj_obs_ptr->object_id << std::endl;
                // std::cout << "New object class " << obj_obs_ptr->object_class << std::endl;

                // insert new object observations pointer
                object_obs_dict[obj_obs_ptr->object_id] = obj_obs_ptr;
            }
            else
            {
                obj_obs_ptr = object_obs_dict.at(bbox.id);
            }

            // for debugging 
            // std::cout << "lost flag " << BoolToString(bbox.lost_flag) << std::endl;

            int object_track_len;

            object_track_len = obj_obs_ptr->zb.size();

            // for debugging 
            // std::cout << "object_track_len " << object_track_len << std::endl;

            // TODO FIXME use the same lost flag for both kitti and unity
            if (!use_unity_dataset_flag)
            {
                // for kitti dataset
                // check whether the object is lost
                // 1. if bbox tracking is lost
                // 2. if object is being tracked for too long
                if (bbox.lost_flag && object_track_len > min_object_feature_track_length)
                {
                    // for debugging 
                    // std::cout << "lost object id " << bbox.id << std::endl;
                    
                    lost_object_ids.push_back(bbox.id);
                }
                else if (object_track_len > max_object_feature_track_length)
                {
                    // for debugging
                    // std::cout << "max_object_feature_track_length " << max_object_feature_track_length << std::endl;
                    // std::cout << "lost object id " << bbox.id << std::endl;
                    // std::cout << "lost object class " << bbox.Class << std::endl;

                    lost_object_ids.push_back(bbox.id);
                }
                else
                {
                    // std::cout << "object track len " << object_track_len << std::endl;
                }
            }
            else
            {
                // for unity dataset
                if (object_track_len > max_object_feature_track_length)
                {
                    // for debugging
                    // std::cout << "max_object_feature_track_length " << object_init_node->max_object_feature_track_length << std::endl;
                    // std::cout << "object_track_len " << object_track_len << std::endl;
                    // std::exit(1);
                    // std::cout << "[Object init Node] Lost object id " << obj_obs_ptr->object_id << std::endl;

                    lost_object_ids.push_back(obj_obs_ptr->object_id);
                }
            }

            // insert timestamps into object observations
            obj_obs_ptr->timestamps[cam_id].emplace_back(bboxes_msg->header.stamp.toSec());

            // for debugging
            // std::cout << "measurement timestamps " << bboxes_msg->header.stamp.toSec() << std::endl;

            Vector4d zb_per_frame;
            zb_per_frame << bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax;
            if (!(zb_per_frame.array().head<2>() < zb_per_frame.array().tail<2>()).all())
            {
                std::stringstream ss;
                ss << "Bad bounding box (maxs are smaller than mins): (xmin, ymin, xmax, ymax): " << zb_per_frame;
                if (zb_per_frame(0) > zb_per_frame(2))
                {
                    std::cout << "[WARN]:" << ss.str() << ". Swamming xmin xmax";
                    std::swap(zb_per_frame(0), zb_per_frame(2));
                }
                else
                {
                    std::swap(zb_per_frame(1), zb_per_frame(3));
                }
                // throw std::runtime_error(ss.str());
            }

            Eigen::Matrix<double, 3, 3> camera_intrinsics;
            cv2eigen(camK, camera_intrinsics);

            assert((zb_per_frame.array().head<2>() < zb_per_frame.array().tail<2>()).all());
            obj_obs_ptr->zb.push_back(normalize_bbox(zb_per_frame, camera_intrinsics));

            draw_bbox(track_image, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax);
        }

        // optimize objects that are lost
        if (lost_object_ids.size() > 0) {
            // for debugging 
            // for (const auto & id : lost_object_ids)
            //     ROS_INFO_STREAM("lost object id " << id);

            do_object_feature_initialization(lost_object_ids);
            remove_used_object_obs(lost_object_ids);
        }

        // publish track image
        publish_track_image(track_image);
        
        visualize_map_only();

    }

    void ObjectInitNode::draw_bbox(cv::Mat &img, double xmin, double ymin, double xmax, double ymax)
    {
        cv::rectangle(img, cv::Point(xmin, ymin),
                      cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), /*line thickness*/ 3);
    }

    void ObjectInitNode::publish_track_image(const cv::Mat &track_image)
    {
        if (trackImagePublisher.getNumSubscribers() < 1)
            return;

        cv_bridge::CvImage cvImage;
        cvImage.header.stamp = ros::Time::now();
        cvImage.header.frame_id = child_frame_id;
        cvImage.encoding = sensor_msgs::image_encodings::BGR8;
        cvImage.image = track_image;

        trackImagePublisher.publish(*cvImage.toImageMsg());

        // ROS_DEBUG("Track image has been published.");
    }

    void ObjectInitNode::visualize_map_only()
    {

        // plot objects
        if (load_gt_object_info_flag)
        {
            publish_gt_objects();
        }

        publish_quadrics();
    }

    void ObjectInitNode::publish_gt_objects()
    {
        visualization_msgs::MarkerArray markers;

        orcvio::vector_eigen<Eigen::MatrixXd> object_position_gt_normalized_vec;
        orcvio::vector_eigen<Eigen::MatrixXd> object_rotation_gt_normalized_vec;

        // set first pose from config file
        R0 = Eigen::Matrix3d::Identity();
        p0 = first_uav_translation_gt;

        const int nobjects = all_object_states_dict.size();

        for (unsigned int i = 0; i < object_position_gt_vec.size(); i++)
        {

            visualization_msgs::Marker marker_bbox;
            marker_bbox.id = nobjects + i;
            marker_bbox.header.frame_id = fixed_frame_id;
            marker_bbox.header.stamp = ros::Time::now();
            marker_bbox.type = visualization_msgs::Marker::CUBE;
            marker_bbox.action = visualization_msgs::Marker::ADD;
            marker_bbox.lifetime = ros::Duration();

            // fixed color for all object categories
            marker_bbox.color.r = 0;
            marker_bbox.color.g = 0;
            marker_bbox.color.b = 1;
            // set transparency
            marker_bbox.color.a = 0.3;

            std::string object_class = object_class_gt_vec.at(i);
            if (object_sizes_gt_dict.find(object_class) != object_sizes_gt_dict.end())
            {
                marker_bbox.scale.x = object_sizes_gt_dict[object_class].at(0);
                marker_bbox.scale.y = object_sizes_gt_dict[object_class].at(1);
                marker_bbox.scale.z = object_sizes_gt_dict[object_class].at(2);
            }
            else
            {
                std::cout << "unkown object class " << object_class_gt_vec.at(i) << std::endl;
            }

            // get first pose of gt and set it as origin
            Eigen::Vector3d object_position_gt;
            if (set_first_pose_as_origin_flag)
                object_position_gt = R0.transpose() * (object_position_gt_vec.at(i) - p0);
            else
                object_position_gt = object_position_gt_vec.at(i);

            Matrix3d object_rotation_gt;
            if (set_first_pose_as_origin_flag)
                object_rotation_gt = R0.transpose() * object_rotation_gt_vec.at(i);
            else
            {
                object_rotation_gt = object_rotation_gt_vec.at(i);
            }

            object_position_gt_normalized_vec.push_back(object_position_gt);
            object_rotation_gt_normalized_vec.push_back(object_rotation_gt);

            // set object pose
            auto pos_object_mesh = object_position_gt;

            marker_bbox.pose.position.x = pos_object_mesh(0, 0);
            marker_bbox.pose.position.y = pos_object_mesh(1, 0);
            marker_bbox.pose.position.z = pos_object_mesh(2, 0);

            Eigen::Quaterniond object_q = Eigen::Quaterniond(object_rotation_gt);
            // normalize the quaternion
            object_q = object_q.normalized();
            marker_bbox.pose.orientation.x = object_q.x();
            marker_bbox.pose.orientation.y = object_q.y();
            marker_bbox.pose.orientation.z = object_q.z();
            marker_bbox.pose.orientation.w = object_q.w();

            markers.markers.push_back(marker_bbox);
        }

        // Publish
        pub_gt_objects.publish(markers);

    }

    void ObjectInitNode::publish_quadrics()
    {
        // for plotting the ellipsoids
        visualization_msgs::MarkerArray markers;

        for (const auto &object : all_object_states_dict)
        {
            // std::cout << "object id " << object.first << std::endl;

            // for ellipsoid
            visualization_msgs::Marker marker;
            marker.header.frame_id = fixed_frame_id;
            marker.header.stamp = ros::Time::now();
            marker.id = object.second.object_id;
            marker.text = object.second.object_class;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.lifetime = ros::Duration();

            // for keypoints
            visualization_msgs::Marker sphere_list;
            sphere_list.header.frame_id = fixed_frame_id;
            sphere_list.header.stamp = ros::Time::now();
            sphere_list.ns = "spheres";
            sphere_list.action = visualization_msgs::Marker::ADD;
            sphere_list.pose.orientation.w = 1.0;
            sphere_list.id = object.second.object_id;
            sphere_list.type = visualization_msgs::Marker::SPHERE_LIST;

            // set color
            // random color
            // color_from_id(marker.id, marker.color);

            // fixed color
            // std::cout << "object class is " << object.second.object_class << std::endl;

            auto class_name_iter = object_standardized_class_name_.find(object.second.object_class);
            if (class_name_iter == object_standardized_class_name_.end()) {
                // unknown object class
                ROS_WARN_STREAM_ONCE_NAMED(
                    "UnknownClass" + object.second.object_class,
                    "Ignoring unknown object class: " <<
                    object.second.object_class);
                continue;
            }
            auto const& class_name = class_name_iter->second;
            marker.color = object_marker_colors_[class_name];

            // // color for semantic keypoints is always green
            // // for all object classes
            // sphere_list.color.r = 0.0f;
            // sphere_list.color.g = 1.0f;
            // sphere_list.color.b = 0.0f;

            // set transparency
            marker.color.a = 1;
            // sphere_list.color.a = 1.0;

            // set shape

            // For the basic shapes, a scale of 1 in all directions means 1 meter on a side
            // ref http://wiki.ros.org/rviz/Tutorials/Markers%3A%20Basic%20Shapes
            const int scale_factor = 5; 

            marker.scale.x = scale_factor * object.second.ellipsoid_shape(0, 0);
            marker.scale.y = scale_factor * object.second.ellipsoid_shape(1, 0);
            marker.scale.z = scale_factor * object.second.ellipsoid_shape(2, 0);

            // // POINTS markers use x and y scale for width/height respectively
            // sphere_list.scale.x = 0.5;
            // sphere_list.scale.y = 0.5;
            // sphere_list.scale.z = 0.5;

            // set pose
            // only allow yaw change
            // Eigen::Matrix<double, 4, 4> pose_SE2 = poseSE32SE2(object.second.object_pose);
            // use the full 6dof estimated pose
            Eigen::Matrix<double, 4, 4> pose_SE2 = object.second.object_pose;

            marker.pose.position.x = pose_SE2(0, 3);
            marker.pose.position.y = pose_SE2(1, 3);
            marker.pose.position.z = pose_SE2(2, 3);

            // convert to quaternion
            Matrix3d R;
            R = pose_SE2.block(0, 0, 3, 3);
            Eigen::Quaterniond q = Eigen::Quaterniond(R);
            // normalize the quaternion
            q = q.normalized();
            marker.pose.orientation.x = q.x();
            marker.pose.orientation.y = q.y();
            marker.pose.orientation.z = q.z();
            marker.pose.orientation.w = q.w();

            markers.markers.push_back(marker);

            // // for plotting the keypoints
            // // std::cout << "kp num is " << object.second.object_keypoints_shape_global_frame.rows() << std::endl;
            // for (int i = 0; i < object.second.object_keypoints_shape_global_frame.rows(); ++i)
            // {
            //     geometry_msgs::Point p;
            //     p.x = object.second.object_keypoints_shape_global_frame(i, 0);
            //     p.y = object.second.object_keypoints_shape_global_frame(i, 1);
            //     p.z = object.second.object_keypoints_shape_global_frame(i, 2);

            //     // cannot plot the points that are too large
            //     if (abs(p.x) < 1e3)
            //         sphere_list.points.push_back(p);
            //     // else
            //     //     std::cout << "px " << p.x << std::endl;
            // }

            // text marker
            visualization_msgs::Marker text_marker(marker);
            text_marker.id = 10 * all_object_states_dict.size() + object.second.object_id;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.pose.position.z = marker.pose.position.z + 0.6 * marker.scale.z;
            text_marker.scale.x = 0;
            text_marker.scale.y = 0;
            text_marker.scale.z = 0.5 * marker.scale.z;
            text_marker.color.a = 1;
            text_marker.text = class_name + ": " + std::to_string(object.second.object_id);
            markers.markers.push_back(text_marker);
        }

        // Publish
        pub_quadrics.publish(markers);
    }

    void ObjectInitNode::save_kps_to_file(const int &object_id, const Eigen::Matrix3Xd &valid_shape_global_frame, std::string filepath_format)
    {
        boost::format boost_filepath_format(filepath_format);
        std::ofstream file((boost_filepath_format % object_id).str());

        // std::cout << "debug file " << file.is_open() << std::endl;

        if (file.is_open())
        {
            file << "object id:\n"
                 << object_id << '\n';
            file << "valid_shape_global_frame:\n"
                 << valid_shape_global_frame.transpose() << '\n';
        }
        // else
        //     std::cout << "cannot open file" << std::endl;
    }

    void ObjectInitNode::do_object_feature_initialization(const std::vector<int> &lost_object_ids)
    {

        std::shared_ptr<ObjectFeature> obj_obs_ptr;

        for (const auto &object_id : lost_object_ids)
        {
            obj_obs_ptr = object_obs_dict.at(object_id);

            // find common timestamps in pose and measurements
            std::vector<double> common_clonetimes;
            obtain_common_timestamps(pose_timestamps, obj_obs_ptr->timestamps[0], common_clonetimes);

            // skip if we don't have enough observations
            if (common_clonetimes.size() > min_object_feature_track_length)
            {
                // only keep bounding box observations whose timestamp is also in pose
                obj_obs_ptr->clean_old_measurements_lite(common_clonetimes);

                // Get all timestamps our clones are at (and thus valid measurement times)
                // Create vector of cloned *CAMERA* poses at each of our clone timesteps
                std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
                // use gt cam poses
                get_times_cam_poses(common_clonetimes, clones_cam);

                bool init_success_flag;
                Eigen::Matrix4d wTq;
                std::shared_ptr<ObjectFeatureInitializer> object_feat_init;

                // choose the appropriate object initializer based on object class
                // we only add valid object class so no need to check object class here 
                auto object_feat_init_iter = all_objects_feat_init_.find(obj_obs_ptr->object_class);
                object_feat_init = object_feat_init_iter->second;

                // only use bounding boxes and ignore keypoints
                std::tie(init_success_flag, wTq) = object_feat_init->single_object_initialization_lite(obj_obs_ptr, clones_cam);

                if (init_success_flag && use_unity_dataset_flag)
                {
                    // reject the pose that is infinite or too close to origin or too far from ground
                    // this seems only applicable to unity, may turn off for kitti
                    Eigen::Vector3d wPq_opt = wTq.block<3, 1>(0, 3);
                    Eigen::Vector2d wPq_opt_xy = wTq.block<2, 1>(0, 3);

                    const double xy_dist_to_origin_threshold = 5;
                    const double z_dist_to_ground_threshold = 2;
                    if (std::isfinite(wTq(0, 0)) && wPq_opt_xy.norm() > xy_dist_to_origin_threshold && abs(wPq_opt(2, 0)) < z_dist_to_ground_threshold)
                        init_success_flag = true;
                    else
                        init_success_flag = false;
                }

                // store the object if initialization is successful
                if (init_success_flag)
                {
                    std::cout << "object init success, id: " << object_id << std::endl;

                    ObjectState object_state;
                    object_state.object_id = object_id;
                    object_state.object_class = obj_obs_ptr->object_class;
                    object_state.object_pose = wTq;
                    object_state.ellipsoid_shape = object_feat_init->getObjectMeanShape();
                    object_state.object_keypoints_shape_global_frame = transform_mean_keypoints_to_global(object_feat_init->getObjectKeypointsMeanShape(), wTq);

                    all_object_states_dict.insert({object_id, object_state});
                    save_object_state_to_file(object_state, common_clonetimes, result_dir_path_object_map + "initial_state_%d.txt");

                    if (do_fine_tune_object_pose_using_lm)
                    {
                        bool success;

                        // object LM lite using bounding box only 
                        success = object_feat_init->single_levenberg_marquardt_lite(*obj_obs_ptr, clones_cam, object_state,
                                                                                use_left_perturbation_flag, use_new_bbox_residual_flag);

                        if (!success)
                        {
                            std::cout << "[Object Init Node] ObjectLM failed"
                                      << "\n";
                        }

                        save_object_state_to_file(object_state, common_clonetimes, result_dir_path_object_map + "after_LM_object_state_%d.txt");
                    }
                    else
                    {
                        std::cout << "[Object Init Node] ObjectLM not used"
                                  << "\n";
                    }
                }
                else
                {
                    // std::cout << "==========object initialization fails==========" << std::endl;
                }
            }
            else
            {
                // std::cout << "==========not enough observations==========" << std::endl;
                // std::cout << "min_object_feature_track_length " << min_object_feature_track_length << std::endl;
                // std::cout << "common_clonetimes " << common_clonetimes.size() << std::endl;
            }
        }
    }

    void ObjectInitNode::remove_used_object_obs(std::vector<int> &lost_object_ids)
    {
        for (const auto &object_id : lost_object_ids)
        {
            object_obs_dict.erase(object_id);
        }
    }

    void ObjectInitNode::color_from_id(const int id, std_msgs::ColorRGBA &color)
    {
        const int SOME_PRIME_NUMBER = 6553;
        int id_dep_num = ((id * SOME_PRIME_NUMBER) % 255); // id dependent number generation
        color.r = (255.0 - id_dep_num) / 255.0;
        color.g = 0.5;
        color.b = (id_dep_num) / 255.0;
    }

    // void ObjectInitNode::obtain_common_timestamps(const std::vector<double>& pose_timestamps_all, const std::vector<double>& mea_timestamps_all, std::vector<double>& common_clonetimes)
    void ObjectInitNode::obtain_common_timestamps(std::vector<double> &pose_timestamps_all, std::vector<double> &mea_timestamps_all, std::vector<double> &common_clonetimes)
    {

        // for debugging
        // std::cout << "pose timestamps " << std::endl;
        // for (const auto& tp : pose_timestamps_all)
        // {
        //     std::cout << tp << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << "measurement timestamps " << std::endl;
        // for (const auto& tp : mea_timestamps_all)
        // {
        //     std::cout << tp << ", ";
        // }
        // std::cout << std::endl;

        // Sort the vector
        // TODO is this necessary?
        // std::sort(pose_timestamps_all.begin(), pose_timestamps_all.end());
        // std::sort(mea_timestamps_all.begin(), mea_timestamps_all.end());

        std::vector<double> v(pose_timestamps_all.size() + mea_timestamps_all.size());
        std::vector<double>::iterator it, st;
        it = set_intersection(pose_timestamps_all.begin(),
                              pose_timestamps_all.end(),
                              mea_timestamps_all.begin(),
                              mea_timestamps_all.end(),
                              v.begin());

        // for debugging
        // std::cout << "common timestamps " << std::endl;
        // for (st = v.begin(); st != it; ++st)
        // {
        //     std::cout << *st << ", ";
        //     common_clonetimes.push_back(*st);
        // }
        // std::cout << std::endl;

        for (st = v.begin(); st != it; ++st)
        {
            common_clonetimes.push_back(*st);
        }

        // common_clonetimes = pose_timestamps_all;
    }

    void ObjectInitNode::get_times_cam_poses(const std::vector<double> &clonetimes, std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> &clones_cam)
    {

        if (clones_imu.size() == 0)
        {
            throw std::runtime_error("No groundtruth poses received, check the groundtruth topic!");
        }

        size_t cam_id = 0;

        // For this camera, create the vector of camera poses
        std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;

        for (const auto &timestamp : clonetimes)
        {

            // Get the position of this clone in the global
            Eigen::Matrix<double, 3, 3> R_ItoG = clones_imu.at(timestamp).Rot_GtoC();
            Eigen::Matrix<double, 3, 1> p_IinG = clones_imu.at(timestamp).pos_CinG();

            Eigen::Matrix<double, 4, 4> T_ItoG;
            T_ItoG.block(0, 0, 3, 3) = R_ItoG;
            T_ItoG.block(0, 3, 3, 1) = p_IinG;

            Eigen::Matrix<double, 4, 4> T_CitoG;
            T_CitoG = T_ItoG * T_CtoI;

            Eigen::Matrix<double, 3, 3> R_GtoCi;
            R_GtoCi = T_CitoG.block(0, 0, 3, 3).transpose();

            Eigen::Matrix<double, 3, 1> p_CioinG;
            p_CioinG = T_CitoG.block(0, 3, 3, 1);

            // Append to our map
            clones_cami.insert({timestamp, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
        }

        // Append to our map
        clones_cam.insert({cam_id, clones_cami});

        return;
    }

} // namespace orcvio
