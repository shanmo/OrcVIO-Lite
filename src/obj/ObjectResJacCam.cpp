#include <unordered_map>
#include <numeric>

#include "sophus/se3.hpp"

#include "orcvio/obj/ObjectLM.h"
#include "orcvio/obj/ObjectResJacCam.h"

#ifndef NDEBUG
constexpr bool DEBUG = false;
#else
constexpr bool DEBUG = false;
#endif

using Eigen::ArrayXi;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixX2d;
using Eigen::MatrixX3d;
using Eigen::MatrixX4d;
using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using std::get;
using std::vector;
using std::cout;
using std::make_shared;

namespace Eigen {
template <typename Scalar, int NR=Eigen::Dynamic, int NC=1>
Scalar* begin(Eigen::Matrix<Scalar, NR, NC>& m) {
    return m.data();
}

template <typename Scalar, int NR=Eigen::Dynamic, int NC=1>
Scalar* end(Eigen::Matrix<Scalar, NR, NC>& m) {
    return m.data() + m.size();
}

template <typename Scalar, int NR=Eigen::Dynamic, int NC=1>
const Scalar* begin(const Eigen::Matrix<Scalar, NR, NC>& m) {
    return m.data();
}

template <typename Scalar, int NR=Eigen::Dynamic, int NC=1>
const Scalar* end(const Eigen::Matrix<Scalar, NR, NC>& m) {
    return m.data() + m.size();
}

} // namespace Eigen

namespace orcvio {

CameraLM::CameraLM(const Eigen::Vector3d& object_mean_shape,
                   const Eigen::Matrix3d& camera_intrinsics,
                   const Eigen::MatrixX3d& object_keypoints_mean,
                   const ObjectFeature& features,
                   const Eigen::VectorXd& residual_weights,
                   const bool use_left_perturbation_flag,
                   const bool use_new_bbox_residual_flag,
                   const double huber_epsilon)
: DenseFunctorSensorState(LMCameraState::DoF(features.zs.size()),
                         ErrorFeatureQuadric::NErrors(features.zs)
                         + ErrorBBoxQuadric::NErrors(features.zs)),
      object_mean_shape_(object_mean_shape),
      camera_intrinsics_(camera_intrinsics),
      object_keypoints_mean_(object_keypoints_mean),
      features_(features),
      residual_functors_({make_shared<ErrorFeatureQuadric>(features.zs,
                                                           camera_intrinsics,
                                                           object_keypoints_mean,
                                                           use_left_perturbation_flag),
                          make_shared<ErrorBBoxQuadric>(features.zs,
                                                       features.zb,
                                                       camera_intrinsics,
                                                       object_keypoints_mean,
                                                       use_left_perturbation_flag,
                                                       use_new_bbox_residual_flag) }),
      residual_weights_(residual_weights),
      huber_(huber_epsilon),
      valid_zs_num_per_frame_(valid_zs_num_per_frame(features.zs))
{
}

LMCameraState LMCameraState::operator+(const LMCameraState::Tangent& dx)
{

    LMCameraState object(wTo_,
                        shape_,
                        semantic_key_pts_,
                        inversePose(frames_wTc_));

    for (int f = 0; f < frames_wTc_.size(); f++) {
        Eigen::Matrix<double, 6, 1> dx_se3 = dx.block<pose_DoF, 1>(f * pose_DoF, 0);
        Sophus::SE3d dx_SE3 = Sophus::SE3d::exp(dx_se3);
        Sophus::SE3d x_p_dx = dx_SE3 * Sophus::SE3d(frames_wTc_[f]);
        object.frames_wTc_[f] = x_p_dx.matrix();
    }

    return object;

}

LMCameraState& LMCameraState::operator+=(const LMCameraState::Tangent& dx) {

    for (int f = 0; f < frames_wTc_.size(); f++) {
        Eigen::Matrix<double, 6, 1> dx_se3 = dx.block<pose_DoF, 1>(f * pose_DoF, 0);
        Sophus::SE3d dx_SE3 = Sophus::SE3d::exp(dx_se3);
        Sophus::SE3d x_p_dx = dx_SE3 * Sophus::SE3d(frames_wTc_[f]);
        frames_wTc_[f] = x_p_dx.matrix();
    }

    return *this;

}

std::ostream& operator<<(std::ostream& o, const LMCameraState& x)
{
    o << "LMObject(wTo=";
    o << x.wTo_ << ", ";
    o << ", deformation=" << x.semantic_key_pts_ << ", shape=" << x.shape_ << ")";
    return o;
}

constexpr int CameraLM::ErrorFeatureQuadric::ErrPerKP;

int CameraLM::ErrorFeatureQuadric::NErrors(const vector_eigen<Eigen::MatrixX2d>& zs) {
  return ErrPerKP * valid_count(zs);
}

CameraLM::ErrorFeatureQuadric::ErrorFeatureQuadric (const vector_eigen<Eigen::MatrixX2d>& zs,
                                                    const Eigen::Matrix3d& camera_intrinsics,
                                                    const Eigen::MatrixX3d& object_keypoints_mean,
                                                    const bool use_left_perturbation_flag)
  : DenseFunctorSensorState(LMCameraState::pose_DoF, NErrors(zs)),
      camera_intrinsics_(camera_intrinsics),
      valid_zs_(filter_valid_zs(zs)),
      valid_indices_(filter_valid_indices(zs)),
      valid_zs_counts_cum_(partial_sum_valid_counts(zs)),
      use_left_perturbation_flag_(use_left_perturbation_flag)
{
}

int CameraLM::ErrorFeatureQuadric::operator() (const LMCameraState& object,
                                           const size_t frameid ,
                                           Eigen::Ref<ValueType> fvec ) const
{
    const MatrixX2d& valid_zs = valid_zs_[frameid];
    const LMSE3 object_frame_wTo = object.get_wTo();
    const std::vector<int>& valid_idx = valid_indices_[frameid];
    auto valid_keypoints = object.getValidSemanticKeyPts(valid_idx);

    // the camera_intrinsics_ is identity since we use normalized 
    // coordinates as input 
    Eigen::Matrix<double, 3, 4> P = camera_intrinsics_ * object.get_wTc(frameid).inverse().topRows<3>();

    Eigen::MatrixX2d uvs = project_object_points(P, object_frame_wTo.matrix(), valid_keypoints);
    Eigen::MatrixX2d errors = uvs - valid_zs;

    // Eigen::MatrixXd errors_transpose = errors.transpose();
    // vectors grouped with by [(u1, v1), (u2, v2), ..., (un, vn)]
    // Eigen::Matrix2Xd errors_transpose = errors.transpose();
    
    Map<Eigen::Matrix2Xd>(fvec.data(), errors.cols(), errors.rows()) = errors.transpose();
    
    return 0;
}

int
CameraLM::ErrorFeatureQuadric::operator() (const InputType& object,
                                           Eigen::Ref<ValueType> fvec_all) const
{
    assert(object.allFinite());
    assert(fvec_all.size() == m_values);
    
    int currrow = 0;
    for (size_t frameid = 0; frameid < valid_zs_.size(); ++frameid) 
    {
        this->operator ()(object, frameid,
                           fvec_all.block(currrow, 0, nerrors(valid_indices_[frameid].size()), 1));
        // note, need to update currrow
        // otherwise residual contains lots of zeros 
        currrow += nerrors(valid_indices_[frameid].size());
    }
    assert(fvec_all.allFinite());
    
    return 0;
}

int CameraLM::ErrorFeatureQuadric::df_test(const InputType& x, Eigen::Ref<JacobianType> fjac) const {
    
    fjac.setZero();
    Eigen::MatrixXd jac_vertical(values(), 6);

    df(x, jac_vertical);

    for (int f = 0; f < x.getFrames_wTc().size(); f++) {

        int err_per_frame = block_start_frame(f+1) - block_start_frame(f);
        
        // change the shape of jacobian 
        // original jacobian matrix stacks all jacobians vertically 
        // new jacobian matrix contains jacobian diagonally 
        fjac.block(
            block_start_frame(f),
            f*6,
            err_per_frame,
            6) =
            jac_vertical.block(
                block_start_frame(f),
                0,
                err_per_frame,
                6);
    }

    return 0;
}

int CameraLM::ErrorFeatureQuadric::df(const LMCameraState& object, const size_t frameid
                                       , Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());
    
    auto valid_idx = valid_indices_[frameid];
    int n = valid_idx.size();
    auto wTo = object.get_wTo().matrix();
    
    // note that in original python version, wTo is used for optical to world 
    // but in cpp version wTo is from object to world, 
    // and wTc is from optical to world 

    // also note here we use cTw, which is from world to optical 
    Eigen::Matrix<double, 4, 4> cTw = object.get_wTc(frameid).inverse();

    auto valid_keypoints = object.getValidSemanticKeyPts(valid_idx);

    Eigen::Matrix<double, 3, 4> P = camera_intrinsics_ * cTw.topRows<3>();

    // jacobian wrt camera state 
    auto p_se_p_cxi = project_object_points_df_camera(P, wTo, cTw, valid_keypoints, use_left_perturbation_flag_);

    assert(p_se_p_cxi.rows() == nerrors(n));
    fjac.block(0, 0, nerrors(n), 6) = p_se_p_cxi.cast<typename JacobianType::Scalar>();

    assert(fjac.allFinite());

    return 0;
}

int
CameraLM::ErrorFeatureQuadric::df(const InputType& object,
                                   Eigen::Ref<JacobianType> fjac_all) const
{

    assert(object.allFinite());
    assert(fjac_all.rows() == m_values);
    assert(fjac_all.cols() == 6);

    fjac_all.setZero();

    for (size_t frameid = 0; frameid < valid_zs_.size(); ++frameid) 
    {
        int nkpts = valid_indices_[frameid].size();

        df(object, frameid,
            fjac_all.block(block_start_frame(frameid), 0, nerrors(nkpts), 6));
    }

    assert(fjac_all.allFinite());

    return 0;

}

constexpr int CameraLM::ErrorBBoxQuadric::ErrPerFrame;

CameraLM::ErrorBBoxQuadric::ErrorBBoxQuadric(const vector_eigen<Eigen::MatrixX2d>& zs,
                                             const vector_eigen<Eigen::Vector4d>& bboxes,
                                             const Eigen::Matrix3d& camera_intrinsics,
                                             const Eigen::MatrixX3d& object_keypoints_mean,
                                             const bool use_left_perturbation_flag,
                                             const bool use_new_bbox_residual_flag)
: DenseFunctorSensorState(LMCameraState::pose_DoF , NErrors(zs)),
      bboxes_(bboxes),
      camera_intrinsics_(camera_intrinsics),
      valid_indices_(filter_valid_indices(zs)),
      valid_zs_counts_cum_(partial_sum_valid_counts(zs)),
      use_left_perturbation_flag_(use_left_perturbation_flag),
      use_new_bbox_residual_flag_(use_new_bbox_residual_flag)
{
    assert(zs.size() == bboxes.size());
    for (auto const& bbox : bboxes) 
    {
      // xmax, ymax must be greater than xmin, ymin
      assert((bbox.array().head<2>() < bbox.array().tail<2>()).all());
    }
}

VectorXd CameraLM::ErrorBBoxQuadric::operator() (const LMCameraState& object, const size_t frameid) const
{
    auto bbox = bboxes_[frameid];
    auto wTo = object.get_wTo().matrix();
    auto object_shape = object.getShape();
    auto Qi = ellipse_from_shape(object_shape);
    auto P = camera_intrinsics_ * (object.get_wTc(frameid).inverse() * wTo).topRows<3>();
    auto Ci = P * Qi * P.transpose();
    auto lines = poly2lineh(bbox2poly(bbox));

    VectorXd ret;
    if (!use_new_bbox_residual_flag_)
    {
        // old bounding box residual 
        // lines @ Ci * lines
        ret = ((lines * Ci).cwiseProduct(lines)).rowwise().sum();
    }
    else 
    {
        // new bounding box residual 
        ret = Eigen::Vector4d::Zero();
        Eigen::Matrix3d U_square = Qi.block<3, 3>(0, 0);

        for (int i = 0; i < lines.rows(); ++i)
        {
            Eigen::Vector3d uline_zb = lines.block<1, 3>(i, 0).transpose();
            Eigen::Vector4d uline_b = P.transpose() * uline_zb;
            Eigen::Vector3d b = uline_b.block<3, 1>(0, 0);
            double b_norm = b.norm();

            // note, plane_orig_dist is -bh in paper 
            double plane_orig_dist = uline_b(3, 0);
            double sqrt_bU2b = sqrt(b.transpose() * U_square * b);
            double sign = plane_orig_dist > 0 ? 1.0 : -1.0;

            // this equation is exactly the same with the one in paper 
            ret(i, 0) = (plane_orig_dist - sign * sqrt_bU2b) / b_norm;
        }
    }

    return ret;
}

int CameraLM::ErrorBBoxQuadric::operator()(const InputType& object,
                                           Eigen::Ref<ValueType> fvec) const
{
    assert(object.allFinite());
    int N = bboxes_.size();
    assert (fvec.rows() == ErrPerFrame*N);
    int currow = 0;
    for (size_t frameid = 0; frameid < bboxes_.size(); ++frameid) 
    {
        auto f = this->operator ()(object, frameid);
        fvec.block(currow, 0, f.rows(), 1) = f;
        currow += f.rows();
    }
    assert(fvec.allFinite());
    return 0;
}

int CameraLM::ErrorBBoxQuadric::df_test(const InputType& x, Eigen::Ref<JacobianType> fjac) const {
    
    fjac.setZero();
    Eigen::MatrixXd jac_vertical(values(), 6);

    df(x, jac_vertical);

    for (int f = 0; f < x.getFrames_wTc().size(); f++) {

        int err_per_frame = block_start_frame(f+1) - block_start_frame(f);
        
        // change the shape of jacobian 
        // original jacobian matrix stacks all jacobians vertically 
        // new jacobian matrix contains jacobian diagonally 
        fjac.block(
            block_start_frame(f),
            f*6,
            err_per_frame,
            6) =
            jac_vertical.block(
                block_start_frame(f),
                0,
                err_per_frame,
                6);
    }
    return 0;
}

int CameraLM::ErrorBBoxQuadric::df(const LMCameraState& object,
                        const size_t frameid, Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite() );

    Eigen::Matrix<double, 4, 4> wTo = object.get_wTo().matrix();
    auto object_shape = object.getShape();
    Eigen::Matrix4d Qi = ellipse_from_shape(object_shape);

    Eigen::Matrix<double, 3, 4> P = camera_intrinsics_ * object.get_wTc(frameid).inverse().topRows<3>();
    Eigen::Matrix<double, 3, 4> P_prime = camera_intrinsics_ * Eigen::Matrix4d::Identity().topRows<3>();

    auto lines = poly2lineh(bbox2poly(bboxes_[frameid]));

    fjac.setZero();
    
    for (int i = 0; i < lines.rows(); ++i) 
    {

        Eigen::Matrix<double, 1, 4> yyw = lines.row(i) * P;
        Eigen::Matrix<double, 1, 4> yyw_prime = lines.row(i) * P_prime;

        auto yyo = yyw * wTo;

        Eigen::Matrix<double, 1, 6> p_be_p_cxi;

        if (!use_new_bbox_residual_flag_)
        {
            // jacobian wrt old bounding box residual 
            if (use_left_perturbation_flag_)
            {
                // using left perturbation 
                Eigen::Matrix<double, 1, 6> p_eb_p_oxi = 2 * yyo * Qi * wTo.transpose() * circledCirc(yyw.transpose().block<4, 1>(0,0)).transpose();

                // jacobian wrt camera pose 
                p_be_p_cxi = -1 * p_eb_p_oxi;
            }
            else 
            {

                // jacobian wrt camera pose
                p_be_p_cxi = -2 * yyo * Qi * wTo.transpose() * object.get_wTc(frameid).inverse().transpose() * circledCirc(yyw_prime.transpose()).transpose();

            }

        }
        else 
        {
            // jacobian wrt new bounding box residual
            Eigen::Vector3d uline_zb = lines.row(i).transpose();
            Eigen::Vector4d uline_b = P.transpose() * uline_zb;
            Eigen::Vector3d b = uline_b.block<3, 1>(0, 0);
            double b_norm = b.norm();

            Eigen::Matrix4Xd p_ulineb_p_Oxi;
            Eigen::Matrix4Xd p_ulineb_p_Cxi;

            if (use_left_perturbation_flag_)
            {
                // jacobian wrt object pose
                p_ulineb_p_Oxi = wTo.transpose() * circledCirc(yyw.transpose().block<4, 1>(0,0)).transpose();
            
                // jacobian wrt camera pose
                p_ulineb_p_Cxi = p_ulineb_p_Oxi;
            }
            else 
            {

                // jacobian wrt camera pose
                p_ulineb_p_Cxi = wTo.transpose() * object.get_wTc(frameid).inverse().transpose() * circledCirc(yyw_prime.transpose()).transpose();

            }

            Eigen::MatrixX4d p_be_p_ulinea = Eigen::MatrixX4d::Zero(1, 4);
            Eigen::MatrixX4d term1a = Eigen::MatrixX4d::Zero(1, 4);
            term1a(0, 3) = 1;
            Eigen::Matrix3d U_square = Qi.block<3, 3>(0, 0);
            Eigen::Matrix4d term2a = Qi;
            term2a(3, 3) = 0;

            double plane_orig_dist = uline_b(3, 0);
            double sign = plane_orig_dist > 0 ? 1.0 : -1.0;
            double sqrt_bU2b = sqrt(b.transpose() * U_square * b);
            p_be_p_ulinea = term1a - sign * (uline_b.transpose() * term2a) / sqrt_bU2b;

            Eigen::Matrix4d p_ulinea_ulineb;
            Eigen::Matrix4d term1b = Eigen::Matrix4d::Identity() / b_norm;
            Eigen::Matrix4d term2b = Eigen::Matrix4d::Identity();
            term2b(3, 3) = 0;
            p_ulinea_ulineb = term1b - (uline_b * uline_b.transpose()) * term2b / pow(b_norm, 3);
            
            // jacobian wrt camera pose
            p_be_p_cxi = -1 * p_be_p_ulinea * p_ulinea_ulineb * p_ulineb_p_Cxi;

        }

        fjac.block<1, 6>(i, 0) = p_be_p_cxi;

    }

    assert(fjac.allFinite());

    return 0;
}

int CameraLM::ErrorBBoxQuadric::df(const InputType& object,
                                   Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());
    assert (fjac.rows() == nerrors());
    assert (fjac.cols() == 6);

    fjac.setZero();

    for (size_t frameid = 0; frameid < numFrames(); ++frameid) 
    {
        df(object, frameid, 
            fjac.block(block_start_frame(frameid), 0, ErrPerFrame, 6));
    }

    assert(fjac.allFinite());
    
    return 0;
}

int CameraLM::operator() (const InputType& object, Eigen::Ref<ValueType> fvec) const
{
    assert(object.allFinite());
    assert(fvec.rows() == block_start_functor(residual_functors_.size()));
    
    fvec.setZero();

    const std::vector<std::string> names{"feature", "bbox", "deform_reg", "quad_reg"};
    for (size_t i = 0; i < residual_functors_.size(); ++i) 
    {
        auto fvec_functor = fvec.block(block_start_functor(i), 0,
                                        residual_functors_[i]->values(), 1);
        (*residual_functors_[i])(object, fvec_functor);

        fvec_functor *= residual_weights_(i);
        
        if (DEBUG)
            cout << "f_" << names[i] << "(x):" << fvec_functor.array().square().sum() << "\n";    
        
    }

    // Apply huber
    huber_(fvec, fvec);

    assert(fvec.allFinite());

    return 0;
}

int CameraLM::df(const InputType& object, Eigen::Ref<JacobianType> fjac) const
{

    assert(object.allFinite());

    // FIXME: This hard coding is bad. Need to find another way.
    assert(fjac.rows() == block_start_functor(2));
    assert(fjac.cols() == object.get_jacobian_dof());

    fjac.setZero();

    const std::vector<std::string> names{"feature", "bbox"};
    for (size_t i = 0; i < 2; ++i) 
    {
        auto fjac_functor = fjac.block(block_start_functor(i), 0,
                                       residual_functors_[i]->values(), object.get_jacobian_dof());
        residual_functors_[i]->df(object, fjac_functor);
        fjac_functor *= residual_weights_(i);
    }

    // we still have to apply the huber weights 
    // even for the jacobians wrt sensor state 
    // since the residual always has huber weights 
    // Apply huber
    ValueType fvec(this->values());
    this->operator()(object, fvec);
    huber_.df(fvec, fjac, fjac);

    assert(fjac.allFinite());
    
    return 0;
}

Eigen::MatrixXd 
CameraLM::get_valid_camera_pose_mat(const InputType& object)
{
    vector_eigen<Eigen::Matrix<double, 6, 1> > valid_camera_pose_vec;

    const int valid_pose_num = object.getFrames_wTc().size();
    Eigen::MatrixXd valid_camera_pose_mat(6, valid_pose_num);

    for (size_t frameid = 0; frameid < valid_pose_num; ++frameid) 
    {
        Eigen::Matrix<double, 4, 4> wTc = object.get_wTc(frameid); 
        Eigen::Matrix<double, 6, 1> se3 = Sophus::SE3d(wTc).log();
        valid_camera_pose_vec.push_back(se3);
    }

    for (int i = 0; i < valid_pose_num; i++)
    {
        valid_camera_pose_mat.col(i) = valid_camera_pose_vec.at(i);
    }

    return valid_camera_pose_mat;
}

int CameraLM::Huber::operator()(const InputType& x, Eigen::Ref<ValueType> fvec) const {
  if (std::isinf(huber_epsilon_)) {
    fvec = x;
    return 0;
  }

  double k = huber_epsilon_;
  double ksq = k*k;
  // NOTE: Be careful x and fvec can be same
  for (int r = 0; r < x.rows(); ++r) 
  {
    if (x(r) < ksq)
      fvec(r) = x(r);
    else
      fvec(r) = 2*k*std::sqrt(x(r)) - ksq;
  }
  return 0;
}

int CameraLM::Huber::df(const InputType& x, const Eigen::MatrixXd& fwdJac,
                        Eigen::Ref<JacobianType> fjac) const {
  if (std::isinf(huber_epsilon_)) {
    fjac = fwdJac;
    return 0;
  }

  double k = huber_epsilon_;
  double ksq = k*k;
  // NOTE: Be careful fwdJac and fjac can be same
  for (int r = 0; r < fwdJac.rows(); ++r) 
  {
    if (x(r) < ksq)
      fjac.row(r) = fwdJac.row(r);
    else
      fjac.row(r) = k/std::sqrt(x(r)) * fwdJac.row(r);
  }
  return 0;
}


std::vector<int> CameraLM::get_zs_num_wrt_timestamps() const {
    return valid_zs_num_per_frame_;
}

} // namespace orcvio