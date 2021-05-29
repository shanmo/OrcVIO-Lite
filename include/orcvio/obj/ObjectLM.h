/* -*- c-basic-offset: 4; */
#ifndef OBJECTLM_H
#define OBJECTLM_H

#include <memory>
#include <numeric>
#include <type_traits>

#include "sophus/se3.hpp"

#include "orcvio/utils/EigenLevenbergMarquardt.h"
#include "orcvio/utils/se3_ops.hpp"
#include "orcvio/obj/ObjectFeature.h"

namespace orcvio {

int valid_count(const Eigen::MatrixX2d& zs);
std::vector<int> valid_zs_num_per_frame(const vector_eigen<Eigen::MatrixX2d>& zs);
int conditional_indexing(const Eigen::MatrixXd& mat, const Eigen::ArrayXi& mask,
                         Eigen::VectorXd& out);
int valid_count(const vector_eigen<Eigen::MatrixX2d>& zs);
Eigen::MatrixX2d filter_valid_zs(const Eigen::MatrixX2d& zs);
vector_eigen<Eigen::MatrixX2d> filter_valid_zs(const vector_eigen<Eigen::MatrixX2d>& zs_all);
std::vector<int> filter_valid_indices(const Eigen::MatrixX2d& zs);
std::vector<std::vector<int>> filter_valid_indices(const vector_eigen<Eigen::MatrixX2d>& zs_all);

Eigen::MatrixX2d bbox2poly(const Eigen::Vector4d& bbox);
Eigen::MatrixX3d poly2lineh(const Eigen::MatrixX2d& points);
Eigen::Matrix4d ellipse_from_shape(const Eigen::Vector3d& object_shape);

/**
 * @brief Adds a 6-DOF(se3) to SE3
 *
 * @param x SE3
 * @param dx Vector6d
 * @return x + dx
 */
Sophus::SE3d operator+(const Sophus::SE3d& x,const Eigen::VectorXd& dx) ;

/**
 * @brief operator <<
 *
 * Helps print SE3
 *
 * @param o
 * @param x
 * @return o
 */
std::ostream& operator<<(std::ostream& o, const Sophus::SE3d& x) ;


/**
* @brief The LMSE3 struct (thin wrapper over SE3) for LM compatibility
*/
 struct LMSE3 : Sophus::SE3d {
   /**
    * @brief Tangent
    * Actually is Vector6d
    */
   typedef Sophus::SE3d::Tangent Tangent;

   /**
    * @brief DoF
    * 6
    */
   constexpr static int DoF = Sophus::SE3d::DoF;

   LMSE3() = default;

   /**
    * Pass constructors to SE3 constructors
    */
   template <typename ... Args>
     LMSE3(Args ... args) : Sophus::SE3d(args ...)
     {}
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

     /**
      * @brief size (DoF) in LMSE3
      *
      * @return
      */
     int size() const {
     return DoF;
   }

   /**
    * @brief Computes norm (diag .* x).norm()
    *
    * @param diag
    * @return norm value
    */
   double scaled_norm(const Tangent& diag) const {
     return diag.cwiseProduct(log()).stableNorm();
   }
 };

/**
* @brief The LMObjectTraj class
*
* Represents A single object with pose, shape and keypoints
*/
 class LMObjectState {
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

   static int DoF(int n_semantic_kpts) {
     return wTo_DoF + ShapeDoF + n_semantic_kpts * KeyptDoF;
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
   LMObjectState() = default;


   /**
    * @brief Constructor when Semantic keypints are provided as homoegeneous coordinates
    * @param wTo (4x4)
    * @param shape (3)
    * @param sem_kps_hom (keypoint number x 4)
    */
   template<typename DefT, typename std::enable_if<DefT::ColsAtCompileTime == 4, int>::type = 0>
     LMObjectState(const wTo_Type& wTo,
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
     LMObjectState(const wTo_Type& wTo,
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
       return DoF(semantic_key_pts_.rows());
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

   /// In the big Vector45d vector where does semantic keypoint block start
   int block_start_sem_keypt(int kpid) const {
       return wTo_DoF + ShapeDoF + kpid * KeyptDoF;
   }

   /// Compute scaled norm given diag vector (Needed for LM compatibility)
   double scaled_norm(const Tangent& diag) const {
      double sum = 0;
      sum  += wTo_.scaled_norm(diag.block<wTo_DoF, 1>(block_start_wTo(), 0));
      sum  += diag.block<ShapeDoF, 1>(block_start_shape(), 0).cwiseProduct(shape_).stableNorm();

      for (int kpid = 0; kpid <  semantic_key_pts_.rows(); ++kpid) {
        sum += semantic_key_pts_.row(kpid)
          .transpose()
          .cwiseProduct(diag.block<KeyptDoF, 1>(block_start_sem_keypt(kpid), 0))
          .stableNorm();
      }
      return sum;
   }

   /// Support addition for this object time (Needed for LM compatibility)
   friend LMObjectState operator+(const LMObjectState& x,
                                            const LMObjectState::Tangent& dx) ;

   /// Support addition for this object time (Needed for LM compatibility)
   LMObjectState& operator+=(const LMObjectState::Tangent& dx) ;
   friend std::ostream& operator<<(std::ostream& o, const LMObjectState& x) ;


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

typedef EigenLevenbergMarquardt::DenseFunctor<double, Eigen::Dynamic, Eigen::Dynamic, LMObjectState>
   DenseFunctorObjectState;

/**
 * Construct a vector of cumulative sums
 */
 std::vector<int> partial_sum_valid_counts(const vector_eigen<Eigen::MatrixX2d>& zs);

/**
 * @brief The ObjectLM to handle error methods for fitting Object pose, mean shape, and Keypoint positions.
 */
class ObjectLM : public DenseFunctorObjectState {
public:

    /// Type aliaces for LM compatibility
    typedef LMObjectState InputType;
    typedef Eigen::VectorXd DiffType;

    ObjectLM(const Eigen::Vector3d& object_mean_shape,
             const Eigen::Matrix3d& camera_intrinsics,
             const Eigen::MatrixX3d& object_keypoints_mean,
             const ObjectFeature& features,
             const vector_eigen<Eigen::Matrix4d> &camposes,
             const Eigen::VectorXd& residual_weights,
             const bool use_left_perturbation_flag,
             const bool use_new_bbox_residual_flag,
             const double huber_epsilon = std::numeric_limits<double>::infinity());

    /**
     * @brief Compute keypoint reprojection error
     */
    struct ErrorFeatureQuadric : DenseFunctorObjectState {
        /// Type aliaces for LM compatibility
        typedef LMObjectState InputType;
        typedef Eigen::VectorXd DiffType;
        constexpr static int ErrPerKP = 2;

        /**
         * @brief Constructor for multi frame observations
         *
         * @param zs
         * @param cTw
         * @param camera_intrinsics
         * @param use_left_perturbation_flag: flag for perturbation 
         */
        ErrorFeatureQuadric(const vector_eigen<Eigen::MatrixX2d>& zs,
                            const vector_eigen<Eigen::Matrix4d>& cTw,
                            const Eigen::Matrix3d& camera_intrinsics,
                            const Eigen::MatrixX3d& object_keypoints_mean,
                            const bool use_left_perturbation_flag);

        /// Computes error vector for the LM
        int operator() (const InputType& object, Eigen::Ref<ValueType> fvec) const;

        /// Computes jacobian for the LM wrt object state 
        int df(const InputType& object, Eigen::Ref<JacobianType> fjac) const;

        /// computes scaled norm for the LM
        double scaled_norm(const DiffType& diag, const InputType& object) const;

        static int NErrors(const vector_eigen<Eigen::MatrixX2d>& zs);

        /**
         * @brief Compute when the frame block starts in the error vector and jacobian matrix row
         *
         * @param frameid
         * @return
         */
        int block_start_frame(int frameid) const {
          // TODO 
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
        int operator() (const LMObjectState& object,
                                    const size_t frameid,
                        Eigen::Ref<ValueType> fvec) const;

        /// helper method for single frame computation
        
        // jacobian wrt object state 
        int df(const LMObjectState& object, const size_t frame_idx, Eigen::Ref<JacobianType> fjac) const;

        const vector_eigen<Eigen::Matrix4d> frames_cTw_;
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
     * @param cTw: Transform from world to camera optical frame, s.t. x_c = cTw x_w
     * @param camera_intrinsics: Camera intrinsic matrix
     * @param use_left_perturbation_flag: flag for perturbation 
     * @param use_new_bbox_residual_flag: flag for new bounding box residual 
     * @return computed error
     */
    struct ErrorBBoxQuadric : DenseFunctorObjectState {
        typedef LMObjectState InputType;
        typedef Eigen::VectorXd DiffType;
        constexpr static int ErrPerFrame = 4;

        ErrorBBoxQuadric(const vector_eigen<Eigen::MatrixX2d> &zs,
                         const vector_eigen<Eigen::Vector4d>& bboxes,
                         const vector_eigen<Eigen::Matrix4d>& cTw,
                         const Eigen::Matrix3d& camera_intrinsics,
                         const Eigen::MatrixX3d& object_keypoints_mean,
                         const bool use_left_perturbation_flag,
                         const bool use_new_bbox_residual_flag) ;
        /// Computes error vector for LM
        int operator() (const InputType& object, Eigen::Ref<ValueType> fvec) const;

        /// Computes jacobian for LM wrt object state 
        int df(const InputType& object, Eigen::Ref<JacobianType> fjac) const;

        /// computes scaled norm for LM
        double scaled_norm(const DiffType& diag, const InputType& object) const;
        static int NErrors(const vector_eigen<Eigen::MatrixX2d>& zs) {
          return ErrPerFrame * zs.size();
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
        ValueType operator() (const LMObjectState& object, const size_t frameid) const;
        
        // get jacobian wrt object state per frame 
        int df(const LMObjectState& object,
                const size_t frameid, Eigen::Ref<JacobianType> fjac) const;

        size_t numFrames() const { return valid_zs_counts_cum_.size() - 1; }
        int nerrors() const {
            return ErrPerFrame * bboxes_.size();
        }

        int block_start_frame(int frameid) const {
            return ErrPerFrame * frameid;
        }
        const vector_eigen<Eigen::Vector4d> bboxes_;
        const vector_eigen<Eigen::Matrix4d> frames_cTw_;
        const Eigen::Matrix3d camera_intrinsics_;
        const std::vector<std::vector<int>> valid_indices_;
        const std::vector<int> valid_zs_counts_cum_;

        const bool use_left_perturbation_flag_;
        const bool use_new_bbox_residual_flag_;

    };


    /**
     * @brief Compute error due to object deformation
     *
     * @param object_keypoints_mean: Object shape without deformation
     *
     * @return computed error
     */
    struct ErrorDeformRegularization : DenseFunctorObjectState {
        typedef LMObjectState InputType;
        typedef Eigen::VectorXd DiffType;
        ErrorDeformRegularization(const vector_eigen<Eigen::MatrixX2d> &zs,
                                  const Eigen::MatrixX3d& object_keypoints_mean);

        int operator()(const InputType& object, Eigen::Ref<ValueType> fvec) const;
        int df(const InputType& object, Eigen::Ref<JacobianType> fjac) const;
        double scaled_norm(const DiffType& diag, const InputType& object) const;
        static int NErrors(const vector_eigen<Eigen::MatrixX2d>& zs,
                           const Eigen::MatrixX3d& object_keypoints_mean) {
          return object_keypoints_mean.rows() * ErrPerKeypt * (partial_sum_valid_counts(zs).size()-1);
        }
    protected:
        int block_start_semkpt(int kpid) const {
            return ErrPerKeypt * kpid;
        }
        constexpr static int ErrPerKeypt = LMObjectState::KeyptDoF;
        int operator()(const LMObjectState& object,
                   const size_t frameid,
                   Eigen::Ref<Eigen::VectorXd> fvec) const;
        int df(const InputType& object, const size_t frameid, Eigen::Ref<JacobianType> fjac) const;
        size_t numFrames() const { return valid_zs_counts_cum_.size() - 1; }
        const Eigen::MatrixX3d object_shape_without_deform_;
        const std::vector<int> valid_zs_counts_cum_;
    };

    /**
     * @brief Compute regularization term due to object deformation from object mean
     *
     * @param feat: Object semantic features
     * @param object_mean_shape: Object mean shape
     *
     * @return computed error
     */
    struct ErrorQuadVRegularization : DenseFunctorObjectState {
        typedef LMObjectState InputType;
        typedef Eigen::VectorXd DiffType;
        ErrorQuadVRegularization(const vector_eigen<Eigen::MatrixX2d> &zs,
                                 const Eigen::MatrixX3d& object_keypoints_mean,
                                 const Eigen::Vector3d& object_mean_shape);
        int operator()(const InputType& objec_mean_shape, Eigen::Ref<ValueType> fvec) const;
        int df(const InputType& objec_mean_shape, Eigen::Ref<JacobianType> fjac) const;
        double scaled_norm(const DiffType& diag, const InputType& wTo) const;
        static int NErrors(const vector_eigen<Eigen::MatrixX2d>& zs) {
          return ErrPerFrame * (partial_sum_valid_counts(zs).size()-1);
        }
    protected:
        constexpr static int ErrPerFrame = LMObjectState::ShapeDoF;
        size_t numFrames() const { return valid_zs_counts_cum_.size() - 1; }
        const Eigen::Vector3d object_mean_shape_;
        const std::vector<int> valid_zs_counts_cum_;
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
    
    // jacobian wrt object state 
    int df(const InputType& object, Eigen::Ref<JacobianType> fjac) const;

    std::vector<int> get_zs_num_wrt_timestamps() const;

    double scaled_norm(const DiffType& diag, const InputType& object) const;

    int block_start_functor(int index) const {
        return std::accumulate(residual_functors_.begin(),
                               residual_functors_.begin() + index,
                               static_cast<int>(0),
                               [](const int sum, const std::shared_ptr<const DenseFunctorObjectState>& res_f) {
                                 return sum + res_f->values();
                               });
    }

protected:

    const Eigen::Vector3d object_mean_shape_;
    const Eigen::Matrix3d camera_intrinsics_;
    const Eigen::MatrixX3d object_keypoints_mean_;
    const ObjectFeature features_;

    const std::vector<std::shared_ptr<const DenseFunctorObjectState> > residual_functors_;
    const Eigen::VectorXd residual_weights_;

    const Huber huber_;

    // for recording the valid keypoint measurements per timestamp 
    const std::vector<int> valid_zs_num_per_frame_;
};


/**
 * @brief bbox2poly
 * @param bbox: (left (xmin), up (ymin), right (xmax), down (ymax))
 * @return
 */
Eigen::MatrixX2d bbox2poly(const Eigen::Vector4d& bbox);


/**
 * @brief poly2lineh
 * @param points: n x 2 consecutive points
 * @return Sequence of lines (n x 3)
 */
Eigen::MatrixX3d poly2lineh(const Eigen::MatrixX2d& points);


/**
 * @brief transform_ellipse
 * @param T : 4x4 Transform
 * @param Q : 4x4 ellipse definition
 * @return
 */
Eigen::Matrix3d transform_ellipse(const Eigen::Matrix4d& T, const Eigen::Matrix4d& Q);


/**
 * @brief LevenbergMarquardtStatusString
 * @param status
 * @return
 */
const std::string& LevenbergMarquardtStatusString(int status);

LMObjectState operator+(const LMObjectState& x, const LMObjectState::Tangent& dx) ;

std::ostream& operator<<(std::ostream& o, const LMObjectState& x) ;

} // end namespace orcvio

#endif // OBJECTLM_H
