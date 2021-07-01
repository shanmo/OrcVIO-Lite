#include <ctime>
#include <math.h>

#include "orcvio/obj/ObjectFeatureInitializer.h"
#include "orcvio/utils/se3_ops.hpp"
#include "orcvio/utils/EigenLevenbergMarquardt.h"
#include "orcvio/feat/FeatureInitializer.h"
#include "orcvio/obj/ObjectState.h"

namespace orcvio
{

    typedef orcvio::FeatureInitializer::ClonePose ClonePose;

    ObjectFeatureInitializer::ObjectFeatureInitializer(orcvio::FeatureInitializerOptions &options,
                                                       const Eigen::Vector3d &object_mean_shape_,
                                                       const Eigen::MatrixX3d &object_keypoints_mean_,
                                                       const Eigen::Matrix3d &camera_intrinsics_,
                                                       const Eigen::Vector4d &residual_weights_)
        : FeatureInitializer(options),
          object_mean_shape(object_mean_shape_),
          object_keypoints_mean(object_keypoints_mean_),
          camera_intrinsics(camera_intrinsics_),
          residual_weights(residual_weights_)
    {
        // use_kabsch_with_ransac_flag = false;
        use_kabsch_with_ransac_flag = true;
    }

    std::tuple<bool, Eigen::Matrix4d> ObjectFeatureInitializer::single_object_initialization(std::shared_ptr<ObjectFeature> obj_obs_ptr, std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM)
    {

        bool init_success_flag = false;
        Eigen::Matrix4d wTq_opt = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d wTq_temp;
        std::vector<int> valid_part_ids;
        std::vector<Eigen::Vector3d> valid_positions;

        // triangulate keypoints
        for (int i = 0; i < object_keypoints_mean.rows(); ++i)
        {

            TriangulationResults triangulation_results;
            std::unordered_map<size_t, std::vector<Eigen::VectorXd>> uvs_norm;
            std::vector<int> valid_frame_ids;
            std::unordered_map<size_t, std::vector<double>> valid_timestamps;

            // for debugging
            // std::cout << "kp id " << i << std::endl;

            if (obj_obs_ptr->zs_to_uvnorm(i, uvs_norm, valid_frame_ids))
            {
                obj_obs_ptr->get_valid_timestamps(valid_frame_ids, valid_timestamps);

                // for debugging
                // if (i == 1)
                // {
                //     for (const auto & uvs : uvs_norm)
                //         for (const auto & uv : uvs.second)
                //             std::cout << "uvs_norm " << uv << std::endl;
                //     // std::exit(0);
                // }

                single_triangulation_common(uvs_norm, valid_timestamps, clonesCAM, triangulation_results);
                valid_part_ids.push_back(i);
                valid_positions.push_back(triangulation_results.p_FinG);

                // for debugging
                // if (i == 1)
                // {
                //     std::cout << "p_FinG " << triangulation_results.p_FinG << std::endl;
                //     std::exit(0);
                // }
                // std::cout << "p_FinG " << triangulation_results.p_FinG.transpose() << std::endl;
            }
        }

        // get object pose
        const int num_valid_pts_threshold = 3;

        int valid_kps_num = static_cast<int>(valid_part_ids.size());

        // for debugging
        // std::cout << "=============valid kps number is=============" << valid_kps_num << std::endl;

        valid_shape_global_frame = Eigen::MatrixXd::Zero(3, valid_kps_num);

        if (valid_kps_num > num_valid_pts_threshold)
        {

            Eigen::Matrix3Xd valid_mean_shape;
            valid_mean_shape = Eigen::MatrixXd::Zero(3, valid_kps_num);

            for (int i = 0; i < valid_kps_num; ++i)
            {
                valid_shape_global_frame.col(i) = valid_positions.at(i).cast<double>();
                valid_mean_shape.col(i) = object_keypoints_mean.row(valid_part_ids.at(i)).transpose();
            }

            // for debugging
            // std::cout << "valid_mean_shape " << valid_mean_shape << std::endl;
            // std::cout << "valid_shape_global_frame " << valid_shape_global_frame << std::endl;

            if (!use_kabsch_with_ransac_flag)
            {
                // use kabsch without ransac
                wTq_opt = findTransform(valid_mean_shape, valid_shape_global_frame);

                // only consider SE2 pose
                // wTq_opt = poseSE32SE2(wTq_opt);

                init_success_flag = true;
            }
            else
            {
                // use kabsch with ransac

                // initialize the combination of kps id
                std::vector<std::vector<int>> valid_kps_comb;
                valid_kps_comb = comb(valid_kps_num, num_valid_pts_threshold);

                int max_num_inliers = -1;
                std::vector<int> inlier_ids_temp;
                std::vector<int> inlier_ids_opt;
                const int max_num_inliers_treshold = num_valid_pts_threshold;

                // for evaluation of Kabsch
                Eigen::MatrixXd valid_shape_global_frame_temp;
                Eigen::Matrix3Xd valid_mean_shape_temp;

                for (auto const &valid_kps_per_iter : valid_kps_comb)
                {

                    valid_shape_global_frame_temp = Eigen::MatrixXd::Zero(3, num_valid_pts_threshold);
                    valid_mean_shape_temp = Eigen::MatrixXd::Zero(3, num_valid_pts_threshold);

                    for (int i = 0; i < num_valid_pts_threshold; ++i)
                    {
                        int kp_id = valid_kps_per_iter.at(i);
                        valid_shape_global_frame_temp.col(i) = valid_shape_global_frame.col(kp_id);
                        valid_mean_shape_temp.col(i) = valid_mean_shape.col(kp_id);
                    }

                    wTq_temp = findTransform(valid_mean_shape_temp, valid_shape_global_frame_temp);
                    int inliers_num = evaluate_kabsch_ransac(wTq_temp, valid_shape_global_frame, valid_mean_shape, inlier_ids_temp);
                    if (max_num_inliers < inliers_num)
                    {
                        max_num_inliers = inliers_num;
                        inlier_ids_opt = inlier_ids_temp;
                    }
                }

                if (max_num_inliers > max_num_inliers_treshold)
                {
                    // re-estimate using all inliers from best model
                    valid_shape_global_frame_temp = Eigen::MatrixXd::Zero(3, inlier_ids_opt.size());
                    valid_mean_shape_temp = Eigen::MatrixXd::Zero(3, inlier_ids_opt.size());

                    for (int i = 0; i < inlier_ids_opt.size(); ++i)
                    {
                        int kp_id = inlier_ids_opt.at(i);
                        valid_shape_global_frame_temp.col(i) = valid_shape_global_frame.col(kp_id).cast<double>();
                        valid_mean_shape_temp.col(i) = valid_mean_shape.col(kp_id).transpose();
                    }

                    wTq_opt = findTransform(valid_mean_shape_temp, valid_shape_global_frame_temp);

                    // only consider SE2 pose
                    // wTq_opt = poseSE32SE2(wTq_opt);

                    init_success_flag = true;
                }
                else
                {
                    init_success_flag = false;

                    // for debugging
                    // std::cout << "[Object init] max_num_inliers " << max_num_inliers << std::endl;
                    // std::cout << "[Object init] max_num_inliers_treshold " << max_num_inliers_treshold << std::endl;
                }
            }
        }

        // for debugging
        // std::cout << "[Object init] wTq from kabsch " << wTq_opt << std::endl;
        // std::exit(0);

        return std::make_tuple(init_success_flag, wTq_opt);
    }

    int evaluate_kabsch_ransac(const Eigen::Matrix4d &wTq, const Eigen::Matrix3Xd &estimated_landmarks_world, const Eigen::Matrix3Xd &MS_kabsch, std::vector<int> &inlier_ids)
    {

        // for debugging
        // std::cout << "check no. valid kps " << estimated_landmarks_world.cols() << std::endl;
        // std::cout << "estimated_landmarks_world " << estimated_landmarks_world << std::endl;
        // std::cout << "MS_kabsch " << MS_kabsch << std::endl;

        inlier_ids = {};
        // const double inlier_dist_threshold = 5;
        const double inlier_dist_threshold = 20;

        Eigen::Vector3d distance;
        int inliers_num = 0;

        Eigen::Matrix3d wRq = wTq.block<3, 3>(0, 0);
        Eigen::Vector3d wPq = wTq.block<3, 1>(0, 3);

        for (int i = 0; i < MS_kabsch.cols(); ++i)
        {
            Eigen::Vector3d mean_kp_global_frame;
            mean_kp_global_frame = wRq * MS_kabsch.col(i) + wPq;
            distance = estimated_landmarks_world.col(i) - mean_kp_global_frame;
            if (distance.norm() < inlier_dist_threshold)
            {
                inlier_ids.push_back(i);
                ++inliers_num;
            }
            else
            {
                // std::cout << "[Object init] distance " << distance.norm() << std::endl;
                // std::cout << "[Object init] inlier_dist_threshold " << inlier_dist_threshold << std::endl;
            }
        }

        return inliers_num;
    }

    std::vector<std::vector<int>> comb(int N, int K)
    {

        std::vector<std::vector<int>> valid_kps_comb;

        std::string bitmask(K, 1); // K leading 1's
        bitmask.resize(N, 0);      // N-K trailing 0's

        // print integers and permute bitmask
        do
        {
            std::vector<int> valid_kps = {};
            for (int i = 0; i < N; ++i) // [0..N-1] integers
            {
                if (bitmask[i])
                {
                    // std::cout << " " << i;
                    valid_kps.push_back(i);
                }
            }
            // std::cout << std::endl;
            valid_kps_comb.push_back(valid_kps);
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

        return valid_kps_comb;
    }

    Eigen::Matrix4d findTransform(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out)
    {

        // Default output
        Eigen::Affine3d A;
        A.linear() = Eigen::Matrix3d::Identity(3, 3);
        A.translation() = Eigen::Vector3d::Zero();

        if (in.cols() != out.cols())
            throw "findTransform(): input data mis-match";

        // std::cout << "in " << in << std::endl;
        // std::cout << "out " << out << std::endl;

        // First find the scale, by finding the ratio of sums of some distances,
        // then bring the datasets to the same scale.
        double dist_in = 0, dist_out = 0;

        for (int col = 0; col < in.cols() - 1; col++)
        {
            dist_in += (in.col(col + 1) - in.col(col)).norm();
            dist_out += (out.col(col + 1) - out.col(col)).norm();
        }

        // if (dist_in <= 0 || dist_out <= 0)
        //     return A;

        double scale = dist_out / dist_in;

        out /= scale;

        // Find the centroids then shift to the origin
        Eigen::Vector3d in_ctr = Eigen::Vector3d::Zero();
        Eigen::Vector3d out_ctr = Eigen::Vector3d::Zero();

        for (int col = 0; col < in.cols(); col++)
        {
            in_ctr += in.col(col);
            out_ctr += out.col(col);
        }

        in_ctr /= in.cols();
        out_ctr /= out.cols();

        for (int col = 0; col < in.cols(); col++)
        {
            in.col(col) -= in_ctr;
            out.col(col) -= out_ctr;
        }

        // SVD
        Eigen::MatrixXd Cov = in * out.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Find the rotation
        double d = (svd.matrixV() * svd.matrixU().transpose()).determinant();

        if (d > 0)
            d = 1.0;
        else
            d = -1.0;

        Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3, 3);
        I(2, 2) = d;
        Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();

        // The final transform
        A.linear() = scale * R;
        A.translation() = scale * (out_ctr - R * in_ctr);

        Eigen::Matrix4d Trans;
        Trans.setIdentity();
        Trans.block<3, 3>(0, 0) = A.linear();
        Trans.block<3, 1>(0, 3) = A.translation();

        // std::cout << "wRq " << A.linear() << std::endl;
        // std::cout << "wPq " << A.translation().transpose() << std::endl;

        return Trans;
    }

    bool ObjectFeatureInitializer::single_levenberg_marquardt(const ObjectFeature &feat,
                                                              const std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                                              ObjectState &objectstate,
                                                              const bool use_left_perturbation_flag,
                                                              const bool use_new_bbox_residual_flag)
    {

        clock_t start = clock(); // measures only CPU time

        constexpr int camid = 0;
        vector_eigen<Eigen::Matrix4d> camposes;
        auto const &frames_clone_cam = clonesCAM.at(camid);
        camposes.reserve(feat.timestamps.at(camid).size());

        for (auto const &time : feat.timestamps.at(camid))
        {
            camposes.push_back(frames_clone_cam.at(time).getTransformGlobalToCam());
        }

        ObjectLM functor(object_mean_shape, Eigen::Matrix3d::Identity(),
                         object_keypoints_mean, feat, camposes,
                         residual_weights,
                         use_left_perturbation_flag,
                         use_new_bbox_residual_flag);
        EigenLevenbergMarquardt::LevenbergMarquardt<ObjectLM> lm(functor);
        lm.setFactor(10);
        LMObjectState object(objectstate.object_pose,
                             object_mean_shape,
                             object_keypoints_mean);

        auto status = lm.minimize(object);
        // check return value
        bool issuccess = (lm.info() == EigenLevenbergMarquardt::ComputationInfo::Success);

        if (!issuccess)
        {
            std::cerr << "[Object LM] LM returned status: " << LevenbergMarquardtStatusString(status) << "\n";
            return false;
        }
        else
        {

            auto const wTo = object.getwTo().matrix();
            objectstate.object_pose = wTo;
            objectstate.ellipsoid_shape = object.getShape();
            objectstate.object_keypoints_shape_global_frame = transform_mean_keypoints_to_global(
                object.getSemanticKeyPts(), wTo);

            // for IMU pose update
            // get the residuals
            fvec_all = Eigen::VectorXd(functor.values());
            functor(object, fvec_all);

            // get the jacobians wrt BOTH object state and camera pose
            // jacobian wrt object state
            fjac_object_state_all = Eigen::MatrixXd(functor.values(), functor.inputs());
            functor.df(object, fjac_object_state_all);

            // jacobian wrt camera state
            LMCameraState sensor_object(objectstate.object_pose,
                                        object_mean_shape,
                                        object_keypoints_mean,
                                        camposes);
            CameraLM sensor_functor(object_mean_shape, Eigen::Matrix3d::Identity(),
                                    object_keypoints_mean, feat,
                                    residual_weights,
                                    use_left_perturbation_flag,
                                    use_new_bbox_residual_flag);

            fjac_sensor_state_all = Eigen::MatrixXd(functor.block_start_functor(2), sensor_object.get_jacobian_dof());
            sensor_functor.df(sensor_object, fjac_sensor_state_all);

            zs_num_wrt_timestamps = functor.get_zs_num_wrt_timestamps();

            // get the timestamps of camera poses for those residuals
            // type of this is std::vector<double>
            object_timestamps = feat.timestamps.at(camid);

            // only keep the feature residual and bounding box residuals
            int total_valid_zs_num = 0;
            for (const auto &num : zs_num_wrt_timestamps)
                total_valid_zs_num += num;

            const int residual_size_to_keep = total_valid_zs_num * 2 + zs_num_wrt_timestamps.size() * 4;
            fvec_all.conservativeResize(residual_size_to_keep, fvec_all.cols());
            fjac_object_state_all.conservativeResize(residual_size_to_keep, fjac_object_state_all.cols());
            fjac_sensor_state_all.conservativeResize(residual_size_to_keep, fjac_sensor_state_all.cols());

            valid_camera_pose_mat = sensor_functor.get_valid_camera_pose_mat(sensor_object);
        }

        clock_t stop = clock();
        std::cerr << "[Object LM] LM success and took: " << static_cast<double>(stop - start) * 1e3 / CLOCKS_PER_SEC << " m secs\n";
        return issuccess;
    }

    bool ObjectFeatureInitializer::single_levenberg_marquardt_lite(const ObjectFeature &feat,
                                                                   const std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                                                   ObjectState &objectstate,
                                                                   const bool use_left_perturbation_flag,
                                                                   const bool use_new_bbox_residual_flag)
    {

        clock_t start = clock(); // measures only CPU time

        constexpr int camid = 0;
        vector_eigen<Eigen::Matrix4d> camposes;
        auto const &frames_clone_cam = clonesCAM.at(camid);
        camposes.reserve(feat.timestamps.at(camid).size());

        for (auto const &time : feat.timestamps.at(camid))
        {
            camposes.push_back(frames_clone_cam.at(time).getTransformGlobalToCam());
        }

        ObjectLMLite functor(object_mean_shape, Eigen::Matrix3d::Identity(),
                         object_keypoints_mean, feat, camposes,
                         residual_weights,
                         use_left_perturbation_flag,
                         use_new_bbox_residual_flag);
        EigenLevenbergMarquardt::LevenbergMarquardt<ObjectLMLite> lm(functor);
        lm.setFactor(10);
        LMObjectStateLite object(objectstate.object_pose,
                             object_mean_shape,
                             object_keypoints_mean);

        auto status = lm.minimize(object);
        // check return value
        bool issuccess = (lm.info() == EigenLevenbergMarquardt::ComputationInfo::Success);

        if (!issuccess)
        {
            std::cerr << "[Object LM] LM returned status: " << LevenbergMarquardtStatusString(status) << "\n";
            return false;
        }
        else
        {
            auto const wTo = object.getwTo().matrix();
            objectstate.object_pose = wTo;
            objectstate.ellipsoid_shape = object.getShape();
            objectstate.object_keypoints_shape_global_frame = transform_mean_keypoints_to_global(
                object.getSemanticKeyPts(), wTo);
        }

        clock_t stop = clock();
        std::cerr << "[Object LM] LM success and took: " << static_cast<double>(stop - start) * 1e3 / CLOCKS_PER_SEC << " m secs\n";
        return issuccess;
    }

    std::tuple<bool, Eigen::Matrix4d> ObjectFeatureInitializer::single_object_initialization_lite(std::shared_ptr<ObjectFeature> obj_obs_ptr, std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM)
    {

        bool init_success_flag = false;
        Eigen::Matrix4d wTq_opt = Eigen::Matrix4d::Identity();

        // bool use_first_bbox_flag = true; 
        bool use_first_bbox_flag = false; 

        // get the first camera pose
        const int cam_id = 0;
        double chosen_timestamp; 
        if (use_first_bbox_flag)
        {
            // use first bbox 
            chosen_timestamp = obj_obs_ptr->timestamps.at(cam_id).at(0);
        }
        else 
        {
            // use last bbox 
            int num_obs = obj_obs_ptr->timestamps.at(cam_id).size();
            chosen_timestamp = obj_obs_ptr->timestamps.at(cam_id).at(num_obs-1);
        }

        ClonePose anchorclone = clonesCAM.at(cam_id).at(chosen_timestamp);
        const Eigen::Matrix<double, 3, 3> R_GtoA = anchorclone.Rot_GtoC();
        const Eigen::Matrix<double, 3, 1> p_AinG = anchorclone.pos_CinG();
        Eigen::Vector3d cPw = -R_GtoA * p_AinG;

        // since we always use normalized coordinates, K is always identity
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

        // in python V = np.diag(v * v)
        Eigen::Vector3d empirical_bbox_scale;
        // default values 
        // empirical_bbox_scale << 1.0, 1.0, 1.0;
        // for ERL demo
        empirical_bbox_scale << .3, .3, .5;
        // for unity 
        // empirical_bbox_scale << 0.6, 0.6, 0.6;

        // Eigen::Vector3d vv = object_mean_shape.cwiseProduct(object_mean_shape);
        Eigen::Vector3d vv = (object_mean_shape.cwiseProduct(empirical_bbox_scale)).cwiseProduct(
            object_mean_shape.cwiseProduct(empirical_bbox_scale));

        auto V = vv.asDiagonal();

        // A = wRq @ V @ wRq.T
        // assume the initial object rotation is identity
        Eigen::Matrix3d wRq = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d A = wRq * V * wRq.transpose();

        // B = K @ oRw
        Eigen::Matrix3d B = K * R_GtoA;

        // bbox_lines = bbox2lines(normalized_bbox)
        Eigen::Vector4d chosen_bbox;
        if (use_first_bbox_flag)
        {
            // use first bbox 
            chosen_bbox = obj_obs_ptr->zb[0];
        }
        else 
        {
            // use last bbox 
            int num_bbox = obj_obs_ptr->zb.size();
            chosen_bbox = obj_obs_ptr->zb[num_bbox-1];
        }

        auto bbox_lines = poly2lineh(bbox2poly(chosen_bbox));
        Eigen::Matrix3d line_sum = Eigen::Matrix3d::Zero();
        double denominator = 0;

        for (int i = 0; i < bbox_lines.rows(); i++)
        {
            auto line = bbox_lines.row(i).transpose();
            // line_sum += line @ line.T
            line_sum += line * line.transpose();
            // denominator += line.T @ B @ A @ B.T @ line
            denominator += line.transpose() * B * A * B.transpose() * line;
        }

        // E = B.T @ line_sum @ B / denominator
        Eigen::Matrix3d E = B.transpose() * line_sum * B / denominator;

        // center = get_bbox_center(normalized_bbox)
        Eigen::Vector2d center;
        center(0, 0) = (chosen_bbox[0] + chosen_bbox[2]) / 2;
        center(1, 0) = (chosen_bbox[1] + chosen_bbox[3]) / 2;

        // b = np.zeros((3, 1))
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        // b[0:2, 0] = center
        b.block<2, 1>(0, 0) = center;
        // b[2, 0] = 1
        b(2, 0) = 1;

        // d = 1 / np.sqrt(b.T @ np.linalg.inv(B).T @ E @ np.linalg.inv(B) @ b)
        double d = 1 / sqrt(b.transpose() * B.inverse().transpose() * E * B.inverse() * b);

        // wPq = np.squeeze(d * np.linalg.inv(B) @ b) - np.squeeze(oRw.T @ oPw)
        Eigen::Vector3d wPq = d * B.inverse() * b - R_GtoA.transpose() * cPw;

        if (wPq.allFinite())
        {
            wTq_opt.block<3, 1>(0, 3) = wPq;

            // only consider SE2 pose
            wTq_opt = poseSE32SE2(wTq_opt);

            init_success_flag = true;
        }

        return std::make_tuple(init_success_flag, wTq_opt);
    }

} // namespace orcvio
