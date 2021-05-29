/* -*- c-basic-offset: 4; */
#ifndef OBJECTRESJACCAM_H
#define OBJECTRESJACCAM_H

#include <memory>
#include <numeric>
#include <type_traits>

#include "sophus/se3.hpp"

#include "orcvio/utils/EigenLevenbergMarquardt.h"
#include "orcvio/utils/se3_ops.hpp"
#include "orcvio/obj/ObjectFeature.h"

namespace orcvio {

/**
* @brief The LMObjectTraj class
*
* Represents A single object with pose, shape and keypoints
*/
 class LMCameraState {
 public:
   typedef int Index;
   /**
    * @brief Tangent
    *
    * Type for dx
    */
   typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Tangent;

   typedef LMSE3 wTo_Type;
   typedef Eigen::Matrix<double, 3, 1> ShapeType;
   typedef Eigen::Matrix<double, Eigen::Dynamic, 3> SemanticKeyPtsType;
   typedef Eigen::Matrix<double, Eigen::Dynamic, 4> HomSemanticKeyPtsType;

   constexpr static int pose_DoF = LMSE3::DoF;

   /**
    * @brief Convert from homogeneous keypints to 3D keypoints
    *
    * @param deformation_hom
    * @return
    */
   static SemanticKeyPtsType toSemanticKeyPts(const HomSemanticKeyPtsType& deformation_hom) {
     Eigen::VectorXd last_col = deformation_hom.block(0,3, deformation_hom.rows(), 1);
     return (deformation_hom.block(0,0, deformation_hom.rows(), 3).array().colwise()
             / last_col.array()).matrix();
   }

   /**
    * @brief Convert from 3D keypoints to Homogeneous keypoints
    * @param deformation
    * @return
    */
   static HomSemanticKeyPtsType toHomSemanticKeyPts(const SemanticKeyPtsType& deformation) {
     HomSemanticKeyPtsType deformation_hom = HomSemanticKeyPtsType::Ones(deformation.rows(), 4);
     deformation_hom.block(0, 0, deformation.rows(), 3) = deformation;
     return deformation_hom;
   }

   /**
    * @brief Default constructor
    */
   LMCameraState() = default;


   /**
    * @brief Constructor when Semantic keypints are provided as homoegeneous coordinates
    * @param wTo (4x4)
    * @param shape (3)
    * @param sem_kps_hom (keypoint number x 4)
    * @param frames_cTw: Transform from world to camera optical frame, s.t. x_c = cTw x_w
    */
   template<typename DefT, typename std::enable_if<DefT::ColsAtCompileTime == 4, int>::type = 0>
     LMCameraState(const wTo_Type& wTo,
                   const ShapeType& shape,
                   const DefT& sem_kps_hom,
                   const vector_eigen<Eigen::Matrix4d> frames_cTw)
     : wTo_(wTo),
     shape_(shape),
     semantic_key_pts_(toSemanticKeyPts(sem_kps_hom)),
     hom_semantic_key_pts_(sem_kps_hom)
     {
      assert(wTo.matrix().allFinite());
      assert(shape.allFinite());
      assert(sem_kps_hom.allFinite());
      assert(allFinite());

      frames_wTc_ = inversePose(frames_cTw);
     }

   /**
    * @brief Constructor when Semantic keypints are provided as non homoegeneous coordinates
    *
    * @param wTo (4x4)
    * @param shape (3)
    * @param sem_kps_inhomo (keypoint number x 3)
    * @param frames_cTw: Vector of Transforms from world to camera optical frame, s.t. x_c = cTw x_w
    */
   template<typename DefT, typename std::enable_if<DefT::ColsAtCompileTime == 3, int>::type = 0>
     LMCameraState(const wTo_Type& wTo,
                   const ShapeType& shape,
                   const DefT& sem_kps_inhomo,
                   const vector_eigen<Eigen::Matrix4d> frames_cTw)
     : wTo_(wTo),
     shape_(shape),
     semantic_key_pts_(sem_kps_inhomo),
     hom_semantic_key_pts_(toHomSemanticKeyPts(sem_kps_inhomo))
     {
      assert(allFinite());

      frames_wTc_ = inversePose(frames_cTw);
     }

   static int DoF(int n_frames) {
     return pose_DoF * n_frames;
   }

   int dof() const {
       return DoF(frames_wTc_.size());
   }

  //  used in object feature initializer 
   int get_jacobian_dof() const {
       return pose_DoF;
   }

   /// Needed for LM compatibility
   int size() const {
       return dof();
   }

   LMSE3 get_wTo() const { return wTo_; }
   ShapeType getShape() const { return shape_; }
   SemanticKeyPtsType getSemanticKeyPts() const { return semantic_key_pts_; }
   HomSemanticKeyPtsType getHomSemanticKeyPts() const { return hom_semantic_key_pts_; }
   const vector_eigen<Eigen::Matrix4d>& getFrames_wTc() const { return frames_wTc_; }
   Eigen::Matrix4d get_wTc(int frameid) const { return frames_wTc_[frameid]; }

   /// Filter out the invalid semantic keypoints
   Eigen::MatrixX4d getValidSemanticKeyPts(const std::vector<int>& valid_idx) const {
      
      Eigen::MatrixX4d valid_keypoints(valid_idx.size(), 4);
      
      int valididx = 0;
      for (auto const& idx : valid_idx)
        valid_keypoints.row(valididx++) = hom_semantic_key_pts_.row(idx);

      return valid_keypoints;
   }

   /// Support addition for this object time (Needed for LM compatibility)
   LMCameraState operator+(const LMCameraState::Tangent& dx);

   /// Support addition for this object time (Needed for LM compatibility)
   LMCameraState& operator+=(const LMCameraState::Tangent& dx);
   friend std::ostream& operator<<(std::ostream& o, const LMCameraState& x) ;


   bool allFinite() const {
     assert(wTo_.matrix().allFinite());
     assert(shape_.allFinite());
     assert(semantic_key_pts_.allFinite());
     return wTo_.matrix().allFinite() && shape_.allFinite() && semantic_key_pts_.allFinite();
   }

   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     protected:

     LMSE3 wTo_{ Eigen::Matrix4d::Identity() };
     ShapeType shape_{ ShapeType::Zero() };
     SemanticKeyPtsType semantic_key_pts_;
     HomSemanticKeyPtsType hom_semantic_key_pts_;

     // This is a part of camera state but kept here so that it can be easily
     // changed to compute numerical jacobian with respect to camera state
     vector_eigen<Eigen::Matrix4d> frames_wTc_;

};

typedef EigenLevenbergMarquardt::DenseFunctor<double, Eigen::Dynamic, Eigen::Dynamic, LMCameraState>
   DenseFunctorSensorState;

/**
 * @brief The CameraLM to handle error methods for fitting Object pose, mean shape, and Keypoint positions.
 */
class CameraLM : public DenseFunctorSensorState {
public:

    /// Type aliaces for LM compatibility
    typedef LMCameraState InputType;
    typedef Eigen::VectorXd DiffType;

    CameraLM(const Eigen::Vector3d& object_mean_shape,
             const Eigen::Matrix3d& camera_intrinsics,
             const Eigen::MatrixX3d& object_keypoints_mean,
             const ObjectFeature& features,
             const Eigen::VectorXd& residual_weights,
             const bool use_left_perturbation_flag,
             const bool use_new_bbox_residual_flag,
             const double huber_epsilon = std::numeric_limits<double>::infinity());

    /**
     * @brief Compute keypoint reprojection error
     */
    struct ErrorFeatureQuadric : DenseFunctorSensorState {
        /// Type aliaces for LM compatibility
        typedef LMCameraState InputType;
        typedef Eigen::VectorXd DiffType;
        constexpr static int ErrPerKP = 2;

        /**
         * @brief Constructor for multi frame observations
         *
         * @param zs
         * @param camera_intrinsics
         * @param use_left_perturbation_flag: flag for perturbation 
         */
        ErrorFeatureQuadric(const vector_eigen<Eigen::MatrixX2d>& zs,
                            const Eigen::Matrix3d& camera_intrinsics,
                            const Eigen::MatrixX3d& object_keypoints_mean,
                            const bool use_left_perturbation_flag);

        /// Computes error vector for the LM
        int operator() (const InputType& object, Eigen::Ref<ValueType> fvec) const;

        /// Computes jacobian for the LM wrt sensor state 
        int df(const InputType& object, Eigen::Ref<JacobianType> fjac) const;
        int df_test(const InputType& object, Eigen::Ref<JacobianType> fjac) const;

        static int NErrors(const vector_eigen<Eigen::MatrixX2d>& zs);

        /**
         * @brief Compute when the frame block starts in the error vector and jacobian matrix row
         *
         * @param frameid
         * @return
         */
        int block_start_frame(int frameid) const {
          // FIXME TODO 
          // should this be valid_zs_counts_cum_[frameid] * 2 ??
          
          // return valid_zs_counts_cum_[frameid];
          return valid_zs_counts_cum_[frameid]*2;
        }

        size_t numFrames() const { return valid_zs_counts_cum_.size() - 1; }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
        /**
         * @brief number of error terms for LM
         *
         * @param nkps
         * @return
         */
        int nerrors(int nkps) const {
            return ErrPerKP * nkps;
        }

        /**
         * @brief Compute keypoint block start in error vector and jacobian matrix row
         *
         * @param kpid
         * @return
         */
        int block_start_sem_kpt(int kpid) const {
            return ErrPerKP * kpid;
        }

        /// helper method for single frame error vector computation computatior
        int operator() (const LMCameraState& object,
                                    const size_t frameid,
                        Eigen::Ref<ValueType> fvec) const;

        /// helper method for single frame computation
        
        // jacobian wrt sensor state 
        int df(const LMCameraState& object, const size_t frame_idx, Eigen::Ref<JacobianType> fjac) const;

        const Eigen::Matrix3d camera_intrinsics_;
        const std::vector<std::vector<int>> valid_indices_;
        const vector_eigen<Eigen::MatrixX2d> valid_zs_;
        const std::vector<int> valid_zs_counts_cum_;

        const bool use_left_perturbation_flag_;
    };

    /**
     * @brief Compute bounding box error
     *
     * @param bbox: Object bounding box
     * @param camera_intrinsics: Camera intrinsic matrix
     * @param use_left_perturbation_flag: flag for perturbation 
     * @param use_new_bbox_residual_flag: flag for new bounding box residual 
     * @return computed error
     */
    struct ErrorBBoxQuadric : DenseFunctorSensorState {
        typedef LMCameraState InputType;
        typedef Eigen::VectorXd DiffType;
        constexpr static int ErrPerFrame = 4;

        ErrorBBoxQuadric(const vector_eigen<Eigen::MatrixX2d> &zs,
                         const vector_eigen<Eigen::Vector4d>& bboxes,
                         const Eigen::Matrix3d& camera_intrinsics,
                         const Eigen::MatrixX3d& object_keypoints_mean,
                         const bool use_left_perturbation_flag,
                         const bool use_new_bbox_residual_flag) ;
        /// Computes error vector for LM
        int operator() (const InputType& object, Eigen::Ref<ValueType> fvec) const;

        /// Computes jacobian for LM wrt sensor state 
        int df(const InputType& object, Eigen::Ref<JacobianType> fjac) const;
        int df_test(const InputType& object, Eigen::Ref<JacobianType> fjac) const;

        static int NErrors(const vector_eigen<Eigen::MatrixX2d>& zs) {
          return ErrPerFrame * zs.size();
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
        ValueType operator() (const LMCameraState& object, const size_t frameid) const;

        // get jacobian wrt sensor state per frame 
        int df(const LMCameraState& object,
                        const size_t frameid, Eigen::Ref<JacobianType> fjac) const;
        
        size_t numFrames() const { return valid_zs_counts_cum_.size() - 1; }
        int nerrors() const {
            return ErrPerFrame * bboxes_.size();
        }

        int block_start_frame(int frameid) const {
            return ErrPerFrame * frameid;
        }
        const vector_eigen<Eigen::Vector4d> bboxes_;
        const Eigen::Matrix3d camera_intrinsics_;
        const std::vector<std::vector<int>> valid_indices_;
        const std::vector<int> valid_zs_counts_cum_;

        const bool use_left_perturbation_flag_;
        const bool use_new_bbox_residual_flag_;

    };

    struct Huber {
        typedef Eigen::VectorXd InputType;
        typedef Eigen::VectorXd ValueType;
        typedef Eigen::MatrixXd JacobianType;
        Huber(double huber_epsilon) : huber_epsilon_(huber_epsilon) {}
        int operator()(const InputType& x, Eigen::Ref<ValueType> fvec) const;
        int df(const InputType& x, const Eigen::MatrixXd& fwdJac,
               Eigen::Ref<JacobianType> fjac) const;
    protected:
        double huber_epsilon_;
    };

    const Eigen::Matrix3d& getCameraIntrinsics() const { return camera_intrinsics_; }
    const Eigen::Vector3d& getObjectMeanShape() const { return object_mean_shape_; }

    int operator()(const InputType& object, Eigen::Ref<ValueType> fvec) const;
    
    // jacobian wrt sensor state 
    int df(const InputType& object, Eigen::Ref<JacobianType> fjac) const;

    std::vector<int> get_zs_num_wrt_timestamps() const;

    int block_start_functor(int index) const {
        return std::accumulate(residual_functors_.begin(),
                               residual_functors_.begin() + index,
                               static_cast<int>(0),
                               [](const int sum, const std::shared_ptr<const DenseFunctorSensorState>& res_f) {
                                 return sum + res_f->values();
                               });
    }

    Eigen::MatrixXd get_valid_camera_pose_mat(const InputType& object);

protected:

    const Eigen::Vector3d object_mean_shape_;
    const Eigen::Matrix3d camera_intrinsics_;
    const Eigen::MatrixX3d object_keypoints_mean_;
    const ObjectFeature features_;

    const std::vector<std::shared_ptr<const DenseFunctorSensorState> > residual_functors_;
    const Eigen::VectorXd residual_weights_;

    const Huber huber_;

    // for recording the valid keypoint measurements per timestamp 
    const std::vector<int> valid_zs_num_per_frame_;
};

LMCameraState operator+(const LMCameraState& x, const LMCameraState::Tangent& dx) ;

std::ostream& operator<<(std::ostream& o, const LMCameraState& x) ;

}

#endif // OBJECTRESJACCAM_H
