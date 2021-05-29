/* -*- c-basic-offset: 4; */
#ifndef OBJECTLMLITE_H
#define OBJECTLMLITE_H

#include <memory>
#include <numeric>
#include <type_traits>

#include "sophus/se3.hpp"

#include "orcvio/utils/EigenLevenbergMarquardt.h"
#include "orcvio/utils/se3_ops.hpp"
#include "orcvio/obj/ObjectFeature.h"

namespace orcvio
{

/**
* @brief The LMObjectTraj class
*
* Represents A single object with pose, shape and keypoints
*/
 class LMObjectStateLite {
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
   constexpr static int wTo_DoF = LMSE3::DoF;
   constexpr static int ShapeDoF = 3;
   constexpr static int KeyptDoF = 3;

   static int DoF() {
     return wTo_DoF + ShapeDoF;
   }

   /**
    * @brief Convert from homogeneous keypints to 3D keypoints
    *
    * @param sem_kps_hom
    * @return
    */
   static SemanticKeyPtsType toSemanticKeyPts(const HomSemanticKeyPtsType& sem_kps_hom) {
     Eigen::VectorXd last_col = sem_kps_hom.block(0,3, sem_kps_hom.rows(), 1);
     return (sem_kps_hom.block(0,0, sem_kps_hom.rows(), 3).array().colwise()
             / last_col.array()).matrix();
   }

   /**
    * @brief Convert from 3D keypoints to Homogeneous keypoints
    * @param sem_kps_hom
    * @return
    */
   static HomSemanticKeyPtsType toHomSemanticKeyPts(const SemanticKeyPtsType& deformation) {
     HomSemanticKeyPtsType sem_kps_hom = HomSemanticKeyPtsType::Ones(deformation.rows(), 4);
     sem_kps_hom.block(0, 0, deformation.rows(), 3) = deformation;
     return sem_kps_hom;
   }

   /**
    * @brief Default constructor
    */
   LMObjectStateLite() = default;


   /**
    * @brief Constructor when Semantic keypints are provided as homoegeneous coordinates
    * @param wTo (4x4)
    * @param shape (3)
    * @param sem_kps_hom (keypoint number x 4)
    */
   template<typename DefT, typename std::enable_if<DefT::ColsAtCompileTime == 4, int>::type = 0>
     LMObjectStateLite(const wTo_Type& wTo,
                   const ShapeType& shape,
                   const DefT& sem_kps_hom)
     : wTo_(wTo),
     shape_(shape),
     semantic_key_pts_(toSemanticKeyPts(sem_kps_hom)),
     hom_semantic_key_pts_(sem_kps_hom)
     {
      assert(wTo.matrix().allFinite());
      assert(shape.allFinite());
      assert(sem_kps_hom.allFinite());
      assert(allFinite());
     }

   /**
    * @brief Constructor when Semantic keypints are provided as non homoegeneous coordinates
    *
    * @param wTo (4x4)
    * @param shape (3)
    * @param sem_kps_inhomo (keypoint number x 3)
    */
   template<typename DefT, typename std::enable_if<DefT::ColsAtCompileTime == 3, int>::type = 0>
     LMObjectStateLite(const wTo_Type& wTo,
                   const ShapeType& shape,
                   const DefT& sem_kps_inhomo)
     : wTo_(wTo),
     shape_(shape),
     semantic_key_pts_(sem_kps_inhomo),
     hom_semantic_key_pts_(toHomSemanticKeyPts(sem_kps_inhomo))
     {
      assert(allFinite());
     }

   int dof() const {
       return DoF();
   }

   /// Needed for LM compatibility
   int size() const {
       return dof();
   }

   LMSE3 getwTo() const { return wTo_; }
   ShapeType getShape() const { return shape_; }
   SemanticKeyPtsType getSemanticKeyPts() const { return semantic_key_pts_; }
   HomSemanticKeyPtsType getHomSemanticKeyPts() const { return hom_semantic_key_pts_; }

   /// Filter out the invalid semantic keypoints
   Eigen::MatrixX4d getValidSemanticKeyPts(const std::vector<int>& valid_idx) const {
      
      Eigen::MatrixX4d valid_keypoints(valid_idx.size(), 4);
      
      int valididx = 0;
      for (auto const& idx : valid_idx)
        valid_keypoints.row(valididx++) = hom_semantic_key_pts_.row(idx);

      return valid_keypoints;
   }

   /// In the big Vector45d vector where does wTo block start
   int block_start_wTo() const {
        return 0;
   }

   /// In the big Vector45d vector where does shape block start
   int block_start_shape() const {
        return wTo_DoF;
   }

   /// Compute scaled norm given diag vector (Needed for LM compatibility)
   double scaled_norm(const Tangent& diag) const {
      double sum = 0;
      sum  += wTo_.scaled_norm(diag.block<wTo_DoF, 1>(block_start_wTo(), 0));
      sum  += diag.block<ShapeDoF, 1>(block_start_shape(), 0).cwiseProduct(shape_).stableNorm();
      return sum;
   }

   /// Support addition for this object time (Needed for LM compatibility)
   friend LMObjectStateLite operator+(const LMObjectStateLite& x,
                                            const LMObjectStateLite::Tangent& dx) ;

   /// Support addition for this object time (Needed for LM compatibility)
   LMObjectStateLite& operator+=(const LMObjectStateLite::Tangent& dx) ;
   friend std::ostream& operator<<(std::ostream& o, const LMObjectStateLite& x) ;


   bool allFinite() const {
     assert(wTo_.matrix().allFinite());
     assert(shape_.allFinite());
     assert(semantic_key_pts_.allFinite());
     return wTo_.matrix().allFinite() && shape_.allFinite() && semantic_key_pts_.allFinite();
   }

   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     private:

     LMSE3 wTo_{ Eigen::Matrix4d::Identity() };
     ShapeType shape_{ ShapeType::Zero() };
     SemanticKeyPtsType semantic_key_pts_;
     HomSemanticKeyPtsType hom_semantic_key_pts_;
 };

 typedef EigenLevenbergMarquardt::DenseFunctor<double, Eigen::Dynamic, Eigen::Dynamic, LMObjectStateLite>
   DenseFunctorObjectStateLite;

/**
* @brief The ObjectLM to handle error methods for fitting Object pose, mean shape, and Keypoint positions.
*/
class ObjectLMLite : public DenseFunctorObjectStateLite
{

    public:
        /// Type aliaces for LM compatibility
        typedef LMObjectStateLite InputType;
        typedef Eigen::VectorXd DiffType;

        ObjectLMLite(const Eigen::Vector3d &object_mean_shape,
                     const Eigen::Matrix3d &camera_intrinsics,
                     const Eigen::MatrixX3d &object_keypoints_mean,
                     const ObjectFeature &features,
                     const vector_eigen<Eigen::Matrix4d> &camposes,
                     const Eigen::VectorXd &residual_weights,
                     const bool use_left_perturbation_flag,
                     const bool use_new_bbox_residual_flag,
                     const double huber_epsilon = std::numeric_limits<double>::infinity());

        /**
     * @brief Compute bounding box error
     *
     * @param bbox: Object bounding box
     * @param cTw: Transform from world to camera optical frame, s.t. x_c = cTw x_w
     * @param camera_intrinsics: Camera intrinsic matrix
     * @param use_left_perturbation_flag: flag for perturbation 
     * @param use_new_bbox_residual_flag: flag for new bounding box residual 
     * @return computed error
     */
        struct ErrorBBoxQuadric : DenseFunctorObjectStateLite
        {
            typedef LMObjectStateLite InputType;
            typedef Eigen::VectorXd DiffType;
            constexpr static int ErrPerFrame = 4;

            ErrorBBoxQuadric(const vector_eigen<Eigen::MatrixX2d> &zs,
                             const vector_eigen<Eigen::Vector4d> &bboxes,
                             const vector_eigen<Eigen::Matrix4d> &cTw,
                             const Eigen::Matrix3d &camera_intrinsics,
                             const Eigen::MatrixX3d &object_keypoints_mean,
                             const bool use_left_perturbation_flag,
                             const bool use_new_bbox_residual_flag);
            /// Computes error vector for LM
            int operator()(const InputType &object, Eigen::Ref<ValueType> fvec) const;

            /// Computes jacobian for LM wrt object state
            int df(const InputType &object, Eigen::Ref<JacobianType> fjac) const;

            /// computes scaled norm for LM
            double scaled_norm(const DiffType &diag, const InputType &object) const;
            static int NErrors(const vector_eigen<Eigen::Vector4d> &zb)
            {
                return ErrPerFrame * zb.size();
            }

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        protected:
            ValueType operator()(const LMObjectStateLite &object, const size_t frameid) const;

            // get jacobian wrt object state per frame
            int df(const LMObjectStateLite &object,
                   const size_t frameid, Eigen::Ref<JacobianType> fjac) const;

            size_t numFrames() const { return bboxes_.size() - 1; }
            int nerrors() const
            {
                return ErrPerFrame * bboxes_.size();
            }

            int block_start_frame(int frameid) const
            {
                return ErrPerFrame * frameid;
            }
            const vector_eigen<Eigen::Vector4d> bboxes_;
            const vector_eigen<Eigen::Matrix4d> frames_cTw_;
            const Eigen::Matrix3d camera_intrinsics_;
            const std::vector<std::vector<int>> valid_indices_;

            const bool use_left_perturbation_flag_;
            const bool use_new_bbox_residual_flag_;
        };

        /**
     * @brief Compute regularization term due to object deformation from object mean
     *
     * @param feat: Object semantic features
     * @param object_mean_shape: Object mean shape
     *
     * @return computed error
     */
        struct ErrorQuadVRegularization : DenseFunctorObjectStateLite
        {
            typedef LMObjectStateLite InputType;
            typedef Eigen::VectorXd DiffType;
            ErrorQuadVRegularization(const vector_eigen<Eigen::Vector4d> &zb,
                                     const Eigen::MatrixX3d &object_keypoints_mean,
                                     const Eigen::Vector3d &object_mean_shape);
            int operator()(const InputType &objec_mean_shape, Eigen::Ref<ValueType> fvec) const;
            int df(const InputType &objec_mean_shape, Eigen::Ref<JacobianType> fjac) const;
            double scaled_norm(const DiffType &diag, const InputType &wTo) const;
            static int NErrors(const vector_eigen<Eigen::Vector4d> &zb)
            {
                // TODO FIXME why -1 
                return ErrPerFrame * (zb.size() - 1);
                // return ErrPerFrame * zb.size();
            }

        protected:
            constexpr static int ErrPerFrame = LMObjectStateLite::ShapeDoF;
            size_t numFrames() const { return zb_.size() - 1; }
            const Eigen::Vector3d object_mean_shape_;
            const vector_eigen<Eigen::Vector4d> zb_;
        };

        struct Huber
        {
            typedef Eigen::VectorXd InputType;
            typedef Eigen::VectorXd ValueType;
            typedef Eigen::MatrixXd JacobianType;
            Huber(double huber_epsilon) : huber_epsilon_(huber_epsilon) {}
            int operator()(const InputType &x, Eigen::Ref<ValueType> fvec) const;
            int df(const InputType &x, const Eigen::MatrixXd &fwdJac,
                   Eigen::Ref<JacobianType> fjac) const;

        protected:
            double huber_epsilon_;
        };

        const Eigen::Matrix3d &getCameraIntrinsics() const { return camera_intrinsics_; }
        const Eigen::Vector3d &getObjectMeanShape() const { return object_mean_shape_; }

        int operator()(const InputType &object, Eigen::Ref<ValueType> fvec) const;

        // jacobian wrt object state
        int df(const InputType &object, Eigen::Ref<JacobianType> fjac) const;

        double scaled_norm(const DiffType &diag, const InputType &object) const;

        int block_start_functor(int index) const
        {
            return std::accumulate(residual_functors_.begin(),
                                   residual_functors_.begin() + index,
                                   static_cast<int>(0),
                                   [](const int sum, const std::shared_ptr<const DenseFunctorObjectStateLite> &res_f) {
                                       return sum + res_f->values();
                                   });
        }

    protected:
        const Eigen::Vector3d object_mean_shape_;
        const Eigen::Matrix3d camera_intrinsics_;
        const Eigen::MatrixX3d object_keypoints_mean_;
        const ObjectFeature features_;

        const std::vector<std::shared_ptr<const DenseFunctorObjectStateLite>> residual_functors_;
        const Eigen::VectorXd residual_weights_;

        const Huber huber_;
};

LMObjectStateLite operator+(const LMObjectStateLite& x, const LMObjectStateLite::Tangent& dx) ;

std::ostream& operator<<(std::ostream& o, const LMObjectStateLite& x) ;

} // end namespace orcvio

#endif // OBJECTLMLITE_H
