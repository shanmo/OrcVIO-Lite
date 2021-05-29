#ifndef FEATUREINITIALIZER_H
#define FEATUREINITIALIZER_H

#include <unordered_map>

#include "orcvio/feat/Feature.h"
#include "orcvio/feat/FeatureInitializerOptions.h"

namespace orcvio {

    /**
     * @brief Class that triangulates feature
     *
     * This class has the functions needed to triangulate and then refine a given 3D feature.
     * As in the standard MSCKF, we know the clones of the camera from propagation and past updates.
     * Thus, we just need to triangulate a feature in 3D with the known poses and then refine it.
     * One should first call the single_triangulation() function afterwhich single_gaussnewton() allows for refinement.
     * Please see the @ref update-featinit page for detailed derivations.
     */
    class FeatureInitializer
    {

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /**
         * @brief Structure which stores pose estimates for use in triangulation
         *
         * - R_GtoC - rotation from global to camera
         * - p_CinG - position of camera in global frame
         */
        struct ClonePose {

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            /// Rotation
            const Eigen::Matrix<double,3,3> _Rot_GtoC;

            /// Position
            const Eigen::Matrix<double,3,1> _pos_CinG;

            /// Constructs pose from rotation and position
            ClonePose(const Eigen::Matrix<double,3,3>& R_GtoC, const Eigen::Matrix<double,3,1>& p_CinG)
                : _Rot_GtoC ( R_GtoC),
              _pos_CinG ( p_CinG)
                {
            }

            /// Constructs from SE3 mat gTc such that x_global = gTc * x_cam
            ClonePose(const Eigen::Matrix<double, 4, 4>& global_T_cam) :
                _Rot_GtoC (global_T_cam.block<3,3>(0, 0).transpose()),
                _pos_CinG ( global_T_cam.topRightCorner<3,1>())
            {
            }

            /// Default constructor
            ClonePose() :
                _Rot_GtoC ( Eigen::Matrix<double,3,3>::Identity()),
                _pos_CinG ( Eigen::Matrix<double,3,1>::Zero())
            {
            }

            /// Accessor for rotation
            const Eigen::Matrix<double,3,3> &Rot_GtoC() const {
                return _Rot_GtoC;
            }

            /// Accessor for position
            const Eigen::Matrix<double,3,1> &pos_CinG() const {
                return _pos_CinG;
            }


            /// Get SE3 transform cTw such that x_cam = cTg * x_global
            const Eigen::Matrix<double, 4, 4> getTransformGlobalToCam() const {
                Eigen::Matrix4d cTw = Eigen::Matrix4d::Identity();
                cTw.topLeftCorner<3,3>() = _Rot_GtoC;
                cTw.topRightCorner<3,1>() = - _Rot_GtoC * _pos_CinG;
                return cTw;
            }

            /// Convert points to camera frame from global frame
            const Eigen::Matrix<double, 3, Eigen::Dynamic> transformGlobalToCam(const Eigen::Matrix<double, 3, Eigen::Dynamic>& points_global) const {
              return _Rot_GtoC * (points_global.colwise() - _pos_CinG);
            }

        }; // end of ClonePose

        /**
         * @brief Default constructor
         * @param options Options for the initializer
         */
        FeatureInitializer(FeatureInitializerOptions &options) : _options(options) {}

        /**
         * @brief Uses a linear triangulation to get initial estimate for the feature
         * @param feat Pointer to feature
         * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of camera in global frame)
         * @return Returns false if it fails to triangulate (based on the thresholds)
         */
        bool single_triangulation(Feature* feat, std::unordered_map<size_t,std::unordered_map<double,ClonePose>> &clonesCAM);

        /**
         * @brief Uses a nonlinear triangulation to refine initial linear estimate of the feature
         * @param feat Pointer to feature
         * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of camera in global frame)
         * @return Returns false if it fails to be optimize (based on the thresholds)
         */
        bool single_gaussnewton(Feature* feat, std::unordered_map<size_t,std::unordered_map<double,ClonePose>> &clonesCAM);

    protected:

        /// Contains options for the initializer process
        FeatureInitializerOptions _options;

        /**
         * @brief Helper function for the gauss newton method that computes error of the given estimate
         * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate
         * @param feat Pointer to the feature
         * @param alpha x/z in anchor
         * @param beta y/z in anchor
         * @param rho 1/z inverse depth
         */
        double compute_error(std::unordered_map<size_t,std::unordered_map<double,ClonePose>> &clonesCAM,Feature* feat,double alpha,double beta,double rho);

    }; // FeatureInitializer class 

    /**
     * @brief Structure which stores results returned by triangulation
     *
     * - p_FinA - feature position in anchor frame 
     * - p_FinG - feature position in global frame 
     * - condA - condition number
     * - anchor_cam_id - camera id of anchor frame 
     * - anchor_clone_timestamp - timestamp of anchor frame 
     * - success_flag - flag that indicates whether triangulation is successful
     */
    struct TriangulationResults {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        double condA;
        Eigen::Vector3d p_FinA;
        Eigen::Vector3d p_FinG;
        size_t anchor_cam_id;
        double anchor_clone_timestamp;
        bool success_flag;

        /// Default constructor
        TriangulationResults() {
            condA = 0;
            p_FinA = Eigen::Vector3d::Zero();
            p_FinG = Eigen::Vector3d::Zero();
            anchor_cam_id = 0;
            anchor_clone_timestamp = 0;
            success_flag = false;
        }

    }; // end of TriangulationResults

    /**
     * @brief Uses a linear triangulation to get initial estimate for a single feature
     * @param uvs_norm UV normalized coordinates that this feature has been seen from (mapped by camera ID)
     * @param timestamps associated with feature observations 
     * @param clonesCAM Map between camera ID to map of timestamp to camera pose estimate (rotation from global to camera, position of camera in global frame)
     * @param triangulation_results a structure that holds the results
     * @return Returns void
     */
    void single_triangulation_common(const std::unordered_map<size_t, std::vector<Eigen::VectorXd>>& uvs_norm,
        const std::unordered_map<size_t, std::vector<double>>& timestamps, 
        std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> &clonesCAM,
        TriangulationResults& triangulation_results);

} // orcvio 

#endif // FEATUREINITIALIZER_H