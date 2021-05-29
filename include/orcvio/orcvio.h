/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Tremendous changes have been made to use it in orcvio

#ifndef ORCVIO_H
#define ORCVIO_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <algorithm>
#include <numeric>

#include "imu_state.h"
#include <orcvio/feat/feature.hpp>
#include <orcvio/feat/feature_msg.h>

#include <fstream>
#include <Eigen/StdVector>

#include <list>

#include "Initializer/FlexibleInitializer.h"

#include "sensors/ImuData.hpp"

namespace orcvio {

class OrcVIO {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    OrcVIO(std::string& config_file_);
    // Disable copy and assign constructor
    OrcVIO(const OrcVIO&) = delete;
    OrcVIO operator=(const OrcVIO&) = delete;

    // Destructor
    ~OrcVIO();

    /*
     * @brief initialize Initialize the VIO.
     */
    bool initialize();

    /*
     * @brief reset Resets the VIO to initial status.
     */
    void reset();

    /*
     * @brief processFeatures
     *    Filtering by imu and feature measurements.
     * @param msg Mono feature measurements.
     * @param imu_msg_buffer Imu msg buffer.
     * @return true if publish msgs
     */
    bool processFeatures(MonoCameraMeasurementPtr msg,
                        std::vector<ImuData>& imu_msg_buffer);

    // Get pose
    Eigen::Isometry3d getTbw();
    Eigen::Isometry3d getTcw();

    // Get velocity
    Eigen::Vector3d getVel();

    // Get pose covariance
    Eigen::Matrix<double, 6, 6> getPpose();

    // Get velocity covariance
    Eigen::Matrix3d getPvel();

    // Get poses of augmented IMU states
    void getSwPoses(vector<Eigen::Isometry3d>& swPoses);

    // Get position of map points
    void getStableMapPointPositions(std::map<orcvio::FeatureIDType,Eigen::Vector3d>& mMapPoints);
    void getActiveMapPointPositions(std::map<orcvio::FeatureIDType,Eigen::Vector3d>& mMapPoints);
    void getMSCKFMapPointPositions(std::map<orcvio::FeatureIDType,Eigen::Vector3d>& mMapPoints);

    typedef boost::shared_ptr<OrcVIO> Ptr;
    typedef boost::shared_ptr<const OrcVIO> ConstPtr;

    void removeLostObjects(Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_f, Eigen::VectorXd &res);
    bool constructObjectResidualJacobians(const Eigen::MatrixXd& jacobian_wrt_sensor_state, const std::vector<double>& object_timestamps, Eigen::MatrixXd& Hx, 
                Eigen::MatrixXd& Hf, Eigen::VectorXd& res, const std::vector<int>& zs_num_wrt_timestamps, const MatrixXd& valid_camera_pose_mat);

    // for unit tests 
    inline void setStateCov(const int& imu_dim, const int& num_clone)
    {
        LEG_DIM = imu_dim;

        const int state_dim = imu_dim + 6 * num_clone;
        state_server.state_cov = Eigen::MatrixXd::Identity(state_dim, state_dim);
    }

    inline void setWinPoseTimestamps(std::vector<double>& timestamps)
    {
        cur_window_timestamps = timestamps;
    }

    // for unit test purpose 
    inline void fixDcamposeDimuposeToI()
    {
        dcampose_dimupose_mat = Eigen::Matrix<double, 6, 6>::Identity();
        dcampose_dimupose_fixed_flag = true; 
    }

    inline void setOutputDir(const std::string& output_dir_) {
      output_dir = output_dir_;
    }

  private:
    /*
     * @brief StateServer Store one IMU states and several
     *    augmented IMU and mappoint states for constructing measurement
     *    model.
     *    and FEJ added by QXC
     */
    struct StateServer {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      IMUState imu_state;

      // first estimated imu state for calculate Jacobian
      IMUState imu_state_FEJ_now;
      IMUState imu_state_FEJ_old;

      // last imu state
      IMUState imu_state_old;

      // augmented imu states and first estimated linearized point
      IMUStateServer imu_states_augment;

      // timestamp compensation: t_imu = t_cam + td, timestamp of camera has to add @td to synchronize with clock of imu
      double td;

      // augmmented feature states
      std::vector<FeatureIDType> feature_states;

      // State covariance matrix
      Eigen::MatrixXd state_cov;

      Eigen::MatrixXd continuous_noise_cov;

      // Imu instrinsics
      Eigen::Matrix3d Ma;
      Eigen::Matrix3d Tg;
      Eigen::Matrix3d As;
      // subset vector for imu instrinsics matrice
      Eigen::Vector3d M1;
      Eigen::Vector3d M2;
      Eigen::Vector3d T1;
      Eigen::Vector3d T2;
      Eigen::Vector3d T3;
      Eigen::Vector3d A1;
      Eigen::Vector3d A2;
      Eigen::Vector3d A3;

      // nuisance imu states
      std::vector<StateIDType> nui_ids;
      IMUStateServer nui_imu_states;
      std::map<StateIDType, std::vector<FeatureIDType>> nui_features;
    };


    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters();

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(const double& time_bound,
        std::vector<ImuData>& imu_msg_buffer);
    void processModel(const double& time,
        const Eigen::Vector3d& m_gyro,
        const Eigen::Vector3d& m_acc);
    void predictNewStateOrcVIO(const double& dt,
        const Vector3d& gyro,
        const Vector3d& acc);
    void predictNewStateLARVIO(const double& dt,
        const Eigen::Vector3d& gyro,
        const Eigen::Vector3d& acc);
        
    // Measurement update
    void stateAugmentation();
    void addFeatureObservations(MonoCameraMeasurementPtr msg);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian_msckf(const StateIDType& state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 2, 6>& H_x,
        Eigen::Matrix<double, 2, 6>& H_e,
        Eigen::Matrix<double, 2, 3>& H_f,
        Eigen::Vector2d& r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian_msckf(const FeatureIDType& feature_id,
        const std::vector<StateIDType>& state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    void measurementUpdate_msckf(const Eigen::MatrixXd& H,
        const Eigen::VectorXd& r);
    bool gatingTestFeature(const Eigen::MatrixXd& H,
        const Eigen::VectorXd&r, const int& dof);
    
    void removeLostFeatures();

    void findRedundantImuStates(
        std::vector<StateIDType>& rm_state_ids);
    void pruneImuStateBuffer();

    // Chi squared test table.
    std::map<int, double> chi_squared_table_feat, chi_squared_table_zupt;
    double chi_square_threshold_feat, chi_square_threshold_zupt;
    int chi_squre_num_threshold; 
    
    // State vector
    StateServer state_server;
    // Maximum number of sliding window states
    int sw_size;

    // Features used
    MapServer map_server;

    // Features pass through the gating test,
    // and are truely used in measurement update.
    // For visualization only.
    MapServer publish_features;

    // Lost SLAM features.
    // For visualization only.
    MapServer lost_slam_features;

    // Active SLAM features.
    // For visualization only.
    MapServer active_slam_features;

    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // Sensors measurement noise parameters
    double imu_gyro_noise;
    double imu_acc_noise;
    double imu_gyro_bias_noise;
    double imu_acc_bias_noise;
    double feature_observation_noise;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // Maximum tracking length for a feature
    int max_track_len;

    // FEJ reset time interval threshold
    double reset_fej_threshold;

    // Config file path
    std::string config_file;

    // Rate of features published by front-end. This variable is
    // used to determine the timing threshold of
    // each iteration of the filter.
    // And decide how many imgs to be used for inclinometer-initializer.
    double features_rate;

    // Rate of imu msg.
    double imu_rate;
    // Threshold for deciding which imu is at the same time with a img frame
    double imu_img_timeTh;

    // QXC: Last m_gyro and m_acc
    Eigen::Vector3d  m_gyro_old;
    Eigen::Vector3d  m_acc_old;

    // Take off stamp
    double take_off_stamp;

    // receive config flag for if applicate FEJ
    bool if_FEJ_config;

    // If applicate FEJ
    bool if_FEJ;

    // least observation number for a valid feature
    int least_Obs_Num;

    // time of last update
    double last_update_time;

    // time of last ZUPT
    double last_ZUPT_time;

    // reset First Estimate Point to current estimate if last update is far away
    void resetFejPoint();

    // store the distances of matched feature coordinates of current and previous image, for ZUPT(zero velocity update)
    std::list<double> coarse_feature_dis;

    // ZUPT relevant
    bool if_ZUPT_valid; // use ZUPT in MSCKF
    bool if_use_feature_zupt_flag; // whether we use feature based method to detect zupt 
    bool if_ZUPT;       // if ZUPT applied in current turn
    
    bool checkZUPTFeat();   // check if need ZUPT using features 
    bool checkZUPTIMU();   // check if need ZUPT using IMU 
    std::vector<ImuData> imu_recent_zupt; // used for ZUPT IMU 
    
    double zupt_max_feature_dis;    // threshold for velocity detection
    void measurementUpdate_ZUPT_vpq();  // measurement function for ZUPT, with measurement of velocity, relative position and relative quaterinion
    double zupt_noise_v;    // diagonal element of zupt_v noise corviance
    double zupt_noise_p;    // diagonal element of zupt_p noise corviance
    double zupt_noise_q;    // diagonal element of zupt_q noise corviance
    Eigen::Vector3d meanSForce;     // store the mean of accelerator between two frames

    // debug log: filtering
    std::ofstream fImuState;
    std::ofstream fTakeOffStamp;

    // debug log: the path for output files
    std::string output_dir;

    // flag for first useful image features
    bool bFirstFeatures;

    // Flexible initializer
    float Static_Duration;      // static scene duration for inclinometer-initializer to utlize, in seconds.
    int Static_Num;             // static img mumber for inclinometer-initializer to utlize, calculate according to @Static_Duration and @features_rate
    boost::shared_ptr<FlexibleInitializer> flexInitPtr;

    // input for initialize td
    double td_input;

    // if estimate td
    bool estimate_td;

    // if estimate extrinsic
    bool estimate_extrin;

    // This function is used to compute the EKF-SLAM measurement Jacobian
    // for a single feature observed at a single camera frame.
    // 3d idp version.
    void measurementJacobian_ekf_3didp(const StateIDType& state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 2, 3>& H_f,
        Eigen::Matrix<double, 2, 6>& H_a,
        Eigen::Matrix<double, 2, 6>& H_x,
        Eigen::Matrix<double, 2, 6>& H_e,
        Eigen::Vector2d& r);
    // This function is used to compute the EKF-SLAM measurement Jacobian
    // for a single feature observed at a single camera frame.
    // 1d idp version.
    void measurementJacobian_ekf_1didp(const StateIDType& state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 2, 1>& H_f,
        Eigen::Matrix<double, 2, 6>& H_a,
        Eigen::Matrix<double, 2, 6>& H_x,
        Eigen::Matrix<double, 2, 6>& H_e,
        Eigen::Vector2d& r);
    // This function computes Jacobian of new EKF-SLAM features
    void featureJacobian_ekf_new(const FeatureIDType& feature_id,
        const std::vector<StateIDType>& state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    // This function computes Jacobian of EKF-SLAM features
    void featureJacobian_ekf(const FeatureIDType& feature_id, 
        Eigen::MatrixXd& H_x, Eigen::Vector2d& r);
    // Measurement update function for hybrid vio
    void measurementUpdate_hybrid(
        const Eigen::MatrixXd& H_ekf_new, const Eigen::VectorXd& r_ekf_new, 
        const Eigen::MatrixXd& H_ekf, const Eigen::VectorXd& r_ekf,
        const Eigen::MatrixXd& H_msckf, const Eigen::VectorXd& r_msckf);
    // Update feature covariance if its anchor is to be removed
    // 3d idp version
    void updateFeatureCov_3didp(const FeatureIDType& feature_id, 
        const StateIDType& old_state_id, const StateIDType& new_state_id);
    // Update feature covariance if its anchor is to be removed
    // 1d idp version
    void updateFeatureCov_1didp(const FeatureIDType& feature_id, 
        const StateIDType& old_state_id, const StateIDType& new_state_id);
    // Update corvariance matrix because of ekf feature lost
    void rmLostFeaturesCov(const std::vector<FeatureIDType>& lost_ids);
    // Maximum number of ekf features in a single grid
    int max_features;

    // For feature points distribution, got to know the camera instrinc and image resolution
    std::vector<int> cam_resolution;
    std::vector<double> cam_intrinsics;
    // Rows and cols number to make grids for feature points distribution.
    // If one of these numbers is zero, then apply the pure msckf.
    // Now max_features will be decided by grid_rows*grid_cols.
    int grid_rows, grid_cols;
    // Boundary of feature measurement coordinate
    double x_min, y_min, x_max, y_max;
    // Width and height of each grid
    double grid_height, grid_width;
    // Map to store the id of current features in state according to their code
    std::map<int, std::vector<FeatureIDType>> grid_map;
    // Update imformation of grid map
    void updateGridMap();
    // Delete redundant features in grid map
    static bool compareByObsNum(
        const std::pair<FeatureIDType,int>& lhs, const std::pair<FeatureIDType,int>& rhs) {
        return lhs.second < rhs.second;  
    } 
    void delRedundantFeatures();

    // Parameters to control if using "1d ipd" or "3d idp for augmented features
    int feature_idp_dim;

    // Function to select the new anchor frame
    StateIDType getNewAnchorId(Feature& feature, const std::vector<StateIDType>& rmIDs);

    // if calibrate imu instrinsic online
    bool calib_imu;

    // Dimension of legacy error state
    int LEG_DIM;

    // Function to calculate error state transition equation
    void calPhiClosedForm(Eigen::MatrixXd& Phi, const double& dtime, 
        const Eigen::Vector3d& f, const Eigen::Vector3d& w, 
        const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, 
        const Eigen::Vector3d& f_old, const Eigen::Vector3d& w_old, 
        const Eigen::Vector3d& acc_old, const Eigen::Vector3d& gyro_old);

    void calPhiEulerMethod(Eigen::MatrixXd& Phi, const double& dtime, 
        const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro);

    // Function to update IMU instrinsic matrices by their sections
    void updateImuMx();

    // If apply Schmidt EKF
    bool use_schmidt;
    // Remove useless nuisance state (ones with no EKF features are anchored at)
    void rmUselessNuisanceState();

    // function to increment the state based on update 
    void incrementState_IMUCam(const VectorXd& delta_x);

    // flags for orcvio 
    int use_left_perturbation_flag;
    int use_closed_form_cov_prop_flag;
    int use_larvio_flag;
    int discard_large_update_flag;

    // for object residual update 
    int use_object_residual_update_cam_pose_flag;

    // for using object residual for camera pose update 
    std::vector<double> cur_window_timestamps; 

    // If using ground truth for initial state;
    bool initial_use_gt;

    double init_state_time;
    Eigen::Vector3d init_gyro_bias;
    Eigen::Vector3d init_acc_bias;
    Eigen::Vector4d init_orientation;
    Eigen::Vector3d init_position;
    Eigen::Vector3d init_velocity;

    // for object sensor jacobian wrt IMU pose 
    Eigen::Matrix<double, 6, 6> dcampose_dimupose_mat;
    bool dcampose_dimupose_fixed_flag;
    std::ofstream debug_img_to_imu_mapping_file_;
};

typedef OrcVIO::Ptr OrcVIOPtr;
typedef OrcVIO::ConstPtr OrcVIOConstPtr;

} // namespace orcvio

#endif
