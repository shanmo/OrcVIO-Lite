#ifndef OBJECTFEATUREINITIALIZER_H
#define OBJECTFEATUREINITIALIZER_H

#include <tuple>
#include <vector>
#include <memory>
#include <algorithm>

#include "orcvio/obj/ObjectFeature.h"
#include "orcvio/obj/ObjectLM.h"
#include "orcvio/obj/ObjectLMLite.h"
#include "orcvio/obj/ObjectResJacCam.h"
#include "orcvio/obj/ObjectState.h"

#include "orcvio/feat/FeatureInitializer.h"

namespace orcvio
{

    /**
 * @brief Adds the object component to the Feature initializer class
 *
 */
    class ObjectFeatureInitializer : public FeatureInitializer
    {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
     * @brief ObjectFeatureInitializer
     * @param options : FeatureInitializerOptions
     * @param object_mean_shape_ : Mean object shape
     * @param camera_intrinsics_  : Camera intrinsics
     */
        ObjectFeatureInitializer(FeatureInitializerOptions &options,
                                 const Eigen::Vector3d &object_mean_shape_,
                                 const Eigen::MatrixX3d &object_keypoints_mean_,
                                 const Eigen::Matrix3d &camera_intrinsics_,
                                 const Eigen::Vector4d &residual_weights_ = Eigen::Vector4d::Ones());

        /**
     * @brief initialize the object keypont positions and object pose 
     * @param obj_obs a structure that holds the object observations 
     * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of camera in global frame)
     * @return Returns a tuple of success flag and object pose wTq
     */
        std::tuple<bool, Eigen::Matrix4d> single_object_initialization(std::shared_ptr<ObjectFeature> obj_obs_ptr,
                                                                       std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM);

        /**
     * @brief initialize the object pose using single frame bounding box for lite version   
     * @param obj_obs a structure that holds the object observations 
     * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of camera in global frame)
     * @return Returns a tuple of success flag and object pose wTq
     */
        std::tuple<bool, Eigen::Matrix4d> single_object_initialization_lite(std::shared_ptr<ObjectFeature> obj_obs_ptr, std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM);

        /**
     * @brief Uses object LM to optimize the object states
     *
     * @param feat Pointer to object feature
     * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of camera in global frame)
     * @param [in/out] objectstate Initial estimate of object state which is updated on fine tuning by LM
     * @param flag to indicate whether to use left perturbation  
     * @param flag to indicate whether to use new bounding box residual 
     * @return Returns false if it fails to be optimize (based on the thresholds)
     */
        bool single_levenberg_marquardt(const ObjectFeature &feat,
                                        const std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                        ObjectState &objectstate,
                                        const bool use_left_perturbation_flag,
                                        const bool use_new_bbox_residual_flag);

        /**
     * @brief Object LM with bounding box only 
     *
     * @param feat Pointer to object feature
     * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of camera in global frame)
     * @param [in/out] objectstate Initial estimate of object state which is updated on fine tuning by LM
     * @param flag to indicate whether to use left perturbation  
     * @param flag to indicate whether to use new bounding box residual 
     * @return Returns false if it fails to be optimize (based on the thresholds)
     */
        bool single_levenberg_marquardt_lite(const ObjectFeature &feat,
                                             const std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                             ObjectState &objectstate,
                                             const bool use_left_perturbation_flag,
                                             const bool use_new_bbox_residual_flag);

        const Eigen::Matrix3d &getCameraIntrinsics() const { return camera_intrinsics; }
        const Eigen::Vector3d &getObjectMeanShape() const { return object_mean_shape; }
        const Eigen::MatrixX3d &getObjectKeypointsMeanShape() const { return object_keypoints_mean; }

        Eigen::Matrix3Xd valid_shape_global_frame;

        // for camera pose update
        // for timestamps
        std::vector<double> object_timestamps;
        // for residual
        Eigen::VectorXd fvec_all;
        // for jacobians
        Eigen::MatrixXd fjac_object_state_all;
        Eigen::MatrixXd fjac_sensor_state_all;
        std::vector<int> zs_num_wrt_timestamps;
        Eigen::MatrixXd valid_camera_pose_mat;

    protected:
        Eigen::Vector3d object_mean_shape;
        // MatrixX3d is a partially dynamic-size (fixed-size) matrix of double (Matrix<double, Dynamic, 3>)
        Eigen::MatrixX3d object_keypoints_mean;
        Eigen::Matrix3d camera_intrinsics;
        Eigen::Vector4d residual_weights;

        bool use_kabsch_with_ransac_flag;
    };

    // Source: http://en.wikipedia.org/wiki/Kabsch_algorithm
    // ref https://github.com/oleg-alexandrov/projects/blob/master/eigen/Kabsch.cpp
    /**
 * @brief Given two sets of 3D points, find the rotation + translation + scale
 *
 * @param The input 3D points are stored as columns, eg 3 x n 
 *
 * @return : Trans: 4 x 4 
 */
    Eigen::Matrix4d findTransform(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out);

    // ref https://stackoverflow.com/questions/12991758/creating-all-possible-k-combinations-of-n-items-in-c
    /**
 * @brief Find the combination of size K from N numbers  
 * @param N: range of number 0 - (N - 1)  
 * @param K: size of the comnbination 
 * @return : A vector of all combinations 
 */
    std::vector<std::vector<int>> comb(int N, int K);

    /**
 * @brief evaluate pose from Kabsch and get number of inliers   
 * @param wTq: object pose   
 * @param estimated_landmarks_world: estimated keypoints in world frame
 * @param MS_kabsch: mean shape of the object
 * @param inlier_ids: holder for ids for inliers 
 * @return : number of inliers 
 */
    int evaluate_kabsch_ransac(const Eigen::Matrix4d &wTq, const Eigen::Matrix3Xd &estimated_landmarks_world, const Eigen::Matrix3Xd &MS_kabsch,
                               std::vector<int> &inlier_ids);

} // namespace orcvio

#endif // OBJECTFEATUREINITIALIZER_H
