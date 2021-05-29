#include <unordered_map>
#include <numeric>

#include "sophus/se3.hpp"

#include "orcvio/obj/ObjectLM.h"
#include "orcvio/utils/se3_ops.hpp"

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

Sophus::SE3d operator+(const Sophus::SE3d& x, const Eigen::VectorXd& dx)
{
    assert(dx.allFinite());

    // note, using left perturbation 
    return Sophus::SE3d::exp(dx) * x;

}

std::ostream& operator<<(std::ostream& o, const Sophus::SE3d& x)
{
    o << "SE3d(" << x.log() << ")";
    return o;
}


int valid_count(const MatrixX2d& zs)
{
    return zs.array().isFinite().cast<int>().rowwise().all().count();
}


std::vector<int> valid_zs_num_per_frame(const vector_eigen<MatrixX2d>& zs)
{
    std::vector<int> valid_zs_count;
    std::transform(zs.begin(), zs.end(),
                    std::back_inserter(valid_zs_count),
                    [](const MatrixX2d& zs) -> int {
                        return valid_count(zs);
                    });
    return valid_zs_count;
}

ObjectLM::ObjectLM(const Eigen::Vector3d& object_mean_shape,
                   const Eigen::Matrix3d& camera_intrinsics,
                   const Eigen::MatrixX3d& object_keypoints_mean,
                   const ObjectFeature& features,
                   const vector_eigen<Eigen::Matrix4d>& camposes,
                   const Eigen::VectorXd& residual_weights,
                   const bool use_left_perturbation_flag,
                   const bool use_new_bbox_residual_flag,
                   const double huber_epsilon)
: DenseFunctorObjectState(LMObjectState::DoF(object_keypoints_mean.rows()),
                         ErrorFeatureQuadric::NErrors(features.zs)
                         + ErrorBBoxQuadric::NErrors(features.zs)
                         + ErrorDeformRegularization::NErrors(features.zs, object_keypoints_mean)
                         + ErrorQuadVRegularization::NErrors(features.zs)),
      object_mean_shape_(object_mean_shape),
      camera_intrinsics_(camera_intrinsics),
      object_keypoints_mean_(object_keypoints_mean),
      features_(features),
      residual_functors_({make_shared<ErrorFeatureQuadric>(features.zs,
                                                           camposes,
                                                           camera_intrinsics,
                                                           object_keypoints_mean,
                                                           use_left_perturbation_flag),
                          make_shared<ErrorBBoxQuadric>(features.zs,
                                                       features.zb,
                                                       camposes,
                                                       camera_intrinsics,
                                                       object_keypoints_mean,
                                                       use_left_perturbation_flag,
                                                       use_new_bbox_residual_flag),
                          make_shared<ErrorDeformRegularization>(features.zs,
                                                                 object_keypoints_mean),
                          make_shared<ErrorQuadVRegularization>(features.zs,
                                                                object_keypoints_mean,
                                                                object_mean_shape) }),
      residual_weights_(residual_weights),
      huber_(huber_epsilon),
      valid_zs_num_per_frame_(valid_zs_num_per_frame(features.zs))
{
}

/**
 * @brief Return part (copy) of the matrix using mask
 * @param mat
 * @param mask
 * @return
 */
int conditional_indexing(const Eigen::MatrixXd& mat, const Eigen::ArrayXi& mask,
                         Eigen::VectorXd& out)
{
    Map<MatrixXd> out_mat(out.data(), mask.count(), mat.cols());
    for (int i = 0, j = 0; i < mat.rows(); ++i) 
    {
        if (mask(i)) {
            out_mat.row(j++) = mat.row(i);
        }
    }
    return 0;
}

std::vector<int> partial_sum_valid_counts(const vector_eigen<MatrixX2d>& zs)
{
    auto valid_zs_count = valid_zs_num_per_frame(zs);
    valid_zs_count.insert(valid_zs_count.begin(), 0);
    std::vector<int> partial_sum_valid_zs;
    std::partial_sum(valid_zs_count.begin(), valid_zs_count.end(), std::back_inserter(partial_sum_valid_zs));
    return partial_sum_valid_zs;
}

int valid_count(const vector_eigen<MatrixX2d>& zs)
{
    auto valid_zs_count = valid_zs_num_per_frame(zs);
    return std::accumulate(valid_zs_count.begin(), valid_zs_count.end(), 0);
}

MatrixX2d filter_valid_zs(const MatrixX2d& zs) {
    MatrixX2d valid_zs(valid_count(zs), 2);
    for (int i = 0, vidx = 0; i < zs.rows(); ++i) 
    {
        if (zs.row(i).allFinite())
            valid_zs.row(vidx++) = zs.row(i);
    }
    return valid_zs;
}

vector_eigen<MatrixX2d> filter_valid_zs(const vector_eigen<MatrixX2d>& zs_all) {
    vector_eigen<MatrixX2d> valid_zs_all;
    for (auto const& zs : zs_all) 
    {
        valid_zs_all.push_back(filter_valid_zs(zs));
    }
    return valid_zs_all;
}

std::vector<int> filter_valid_indices(const MatrixX2d& zs) {
    std::vector<int> valid_idx;
    for (int i = 0; i < zs.rows(); ++i) 
    {
        if (zs.row(i).allFinite())
            valid_idx.push_back(i);
    }
    return valid_idx;
}


std::vector<std::vector<int>> filter_valid_indices(const vector_eigen<MatrixX2d>& zs_all) {
    vector<std::vector<int>> valid_idx_all;
    for (auto const& zs : zs_all) 
    {
        valid_idx_all.push_back(filter_valid_indices(zs));
    }
    return valid_idx_all;
}


LMObjectState operator+(const LMObjectState& x, const LMObjectState::Tangent& dx)
{
    LMSE3 sum_frames_wTo(x.wTo_);
    sum_frames_wTo = x.wTo_ + dx.block<LMObjectState::wTo_DoF, 1>(x.block_start_wTo(), 0);

    auto sum_shape = x.shape_ + dx.block<LMObjectState::ShapeDoF, 1>(x.block_start_shape(), 0);

    auto sum_deformation = x.semantic_key_pts_ +
        Map<const MatrixXd>(dx.data() + x.block_start_sem_keypt(0),
                            x.semantic_key_pts_.cols(),
                            x.semantic_key_pts_.rows()).transpose();

    LMObjectState object(sum_frames_wTo,
                         sum_shape,
                         sum_deformation);
    return object;
}

LMObjectState& LMObjectState::operator+=(const LMObjectState::Tangent& dx) {
    wTo_ = wTo_ + dx.block<LMObjectState::wTo_DoF, 1>(block_start_wTo(), 0);

    shape_ = shape_ + dx.block<LMObjectState::ShapeDoF, 1>(block_start_shape(), 0);

    semantic_key_pts_ = semantic_key_pts_ +
        Map<const MatrixXd>(dx.data() + block_start_sem_keypt(0),
                            semantic_key_pts_.cols(),
                            semantic_key_pts_.rows()).transpose();
    hom_semantic_key_pts_ = toHomSemanticKeyPts(semantic_key_pts_);
    return *this;
}

std::ostream& operator<<(std::ostream& o, const LMObjectState& x)
{
    o << "LMObject(wTo=";
    o << x.wTo_ << ", ";
    o << ", deformation=" << x.semantic_key_pts_ << ", shape=" << x.shape_ << ")";
    return o;
}

constexpr int ObjectLM::ErrorFeatureQuadric::ErrPerKP;

int ObjectLM::ErrorFeatureQuadric::NErrors(const vector_eigen<Eigen::MatrixX2d>& zs) {
  return ErrPerKP * valid_count(zs);
}

ObjectLM::ErrorFeatureQuadric::ErrorFeatureQuadric (const vector_eigen<Eigen::MatrixX2d>& zs,
                                                    const vector_eigen<Eigen::Matrix4d> &cTw,
                                                    const Eigen::Matrix3d& camera_intrinsics,
                                                    const Eigen::MatrixX3d& object_keypoints_mean,
                                                    const bool use_left_perturbation_flag)
  : DenseFunctorObjectState(LMObjectState::DoF(object_keypoints_mean.rows()), NErrors(zs)),
      frames_cTw_(cTw),
      camera_intrinsics_(camera_intrinsics),
      valid_zs_(filter_valid_zs(zs)),
      valid_indices_(filter_valid_indices(zs)),
      valid_zs_counts_cum_(partial_sum_valid_counts(zs)),
      use_left_perturbation_flag_(use_left_perturbation_flag)
{
    assert(zs.size() == frames_cTw_.size());
}

int ObjectLM::ErrorFeatureQuadric::operator() (const LMObjectState& object,
                                           const size_t frameid ,
                                           Eigen::Ref<ValueType> fvec ) const
{
    const MatrixX2d& valid_zs = valid_zs_[frameid];
    const LMSE3 object_frame_wTo = object.getwTo();
    const std::vector<int>& valid_idx = valid_indices_[frameid];
    auto valid_keypoints = object.getValidSemanticKeyPts(valid_idx);

    // the camera_intrinsics_ is identity since we use normalized 
    // coordinates as input 
    Eigen::Matrix<double, 3, 4> P = camera_intrinsics_ * frames_cTw_[frameid].topRows<3>();

    Eigen::MatrixX2d uvs = project_object_points(P, object_frame_wTo.matrix(), valid_keypoints);
    Eigen::MatrixX2d errors = uvs - valid_zs;

    // Eigen::MatrixXd errors_transpose = errors.transpose();
    // vectors grouped with by [(u1, v1), (u2, v2), ..., (un, vn)]
    // Eigen::Matrix2Xd errors_transpose = errors.transpose();
    
    Map<Eigen::Matrix2Xd>(fvec.data(), errors.cols(), errors.rows()) = errors.transpose();
    
    return 0;
}

int
ObjectLM::ErrorFeatureQuadric::operator() (const InputType& object,
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

int ObjectLM::ErrorFeatureQuadric::df(const LMObjectState& object, const size_t frameid
                                       , Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());
    
    auto valid_idx = valid_indices_[frameid];
    int n = valid_idx.size();
    auto wTo = object.getwTo().matrix();
    auto valid_keypoints = object.getValidSemanticKeyPts(valid_idx);

    auto P = camera_intrinsics_ * frames_cTw_[frameid].topRows<3>();

    // jacobian wrt object pose 
    // this depends on which perturbation is used 
    auto fjac_wTo = project_object_points_df_object(P, wTo, valid_keypoints, use_left_perturbation_flag_);
    assert(fjac_wTo.rows() == nerrors(n));
    fjac.block(0, object.block_start_wTo(), nerrors(n), fjac_wTo.cols()) = fjac_wTo.cast<typename JacobianType::Scalar>();
    
    // jacobian wrt object keypoints 
    Eigen::Matrix4Xd X_w = object.getHomSemanticKeyPts().transpose();
    int valid_kp_idx = 0;
    for (auto const kpid : valid_idx) 
    {
        Eigen::Vector3d Mc = P * wTo * X_w.col(kpid).topLeftCorner<4, 1>();
        Eigen::Matrix<double, 2, 3> J_uvs_Mw = project_image_df(Mc) * P * wTo.leftCols<3>();
        fjac.block<ErrPerKP, LMObjectState::KeyptDoF>(block_start_sem_kpt(valid_kp_idx++), object.block_start_sem_keypt(kpid)) = J_uvs_Mw;
    }

    assert(fjac.allFinite());

    return 0;
}

int
ObjectLM::ErrorFeatureQuadric::df(const InputType& object,
                                   Eigen::Ref<JacobianType> fjac_all) const
{
    assert(object.allFinite());
    assert(fjac_all.rows() == m_values);
    assert(fjac_all.cols() == m_inputs);

    fjac_all.setZero();

    for (size_t frameid = 0; frameid < valid_zs_.size(); ++frameid) 
    {
        int nkpts = valid_indices_[frameid].size();
        df(object, frameid,
            fjac_all.block(block_start_frame(frameid), 0, nerrors(nkpts), object.dof()));
    }

    assert(fjac_all.allFinite());

    return 0;
}

double
ObjectLM::ErrorFeatureQuadric::scaled_norm (const DiffType& diag,
                                            const InputType& object) const
{
    return object.scaled_norm(diag);
}

Eigen::MatrixX2d bbox2poly(const Eigen::Vector4d& bbox)
{
    auto xmin = bbox(0);
    auto ymin = bbox(1);
    auto xmax = bbox(2);
    auto ymax = bbox(3);
    Eigen::MatrixX2d points(4, 2);
    points << xmin, ymin,
        xmax, ymin,
        xmax, ymax,
        xmin, ymax;
    return points;
}

Eigen::MatrixX3d poly2lineh(const Eigen::MatrixX2d& points)
{
    Eigen::MatrixX3d lines_hom(points.rows(), 3);
    for (int i = 0; i < points.rows(); ++i) 
    {
        Eigen::Vector3d a{points(i, 0), points(i, 1), 1};
        int ip1 = (i + 1) % points.rows();
        Eigen::Vector3d b{points(ip1, 0), points(ip1, 1), 1};
        lines_hom.row(i) = a.cross(b);
    }
    return lines_hom;
}

Eigen::Matrix4d ellipse_from_shape(const Eigen::Vector3d& object_shape)
{
    Eigen::Vector3d vsq = object_shape.array().square().matrix();
    Eigen::Vector4d vsqh;
    vsqh << vsq, -1;
    Eigen::Matrix4d Qi = vsqh.asDiagonal();
    return Qi;
}

constexpr int ObjectLM::ErrorBBoxQuadric::ErrPerFrame;

ObjectLM::ErrorBBoxQuadric::ErrorBBoxQuadric(const vector_eigen<Eigen::MatrixX2d>& zs,
                                             const vector_eigen<Eigen::Vector4d>& bboxes,
                                             const vector_eigen<Eigen::Matrix4d> &cTw,
                                             const Eigen::Matrix3d& camera_intrinsics,
                                             const Eigen::MatrixX3d& object_keypoints_mean,
                                             const bool use_left_perturbation_flag,
                                             const bool use_new_bbox_residual_flag)
: DenseFunctorObjectState(LMObjectState::DoF(object_keypoints_mean.rows()) , NErrors(zs)),
      bboxes_(bboxes),
      frames_cTw_(cTw),
      camera_intrinsics_(camera_intrinsics),
      valid_indices_(filter_valid_indices(zs)),
      valid_zs_counts_cum_(partial_sum_valid_counts(zs)),
      use_left_perturbation_flag_(use_left_perturbation_flag),
      use_new_bbox_residual_flag_(use_new_bbox_residual_flag)
{
    assert(zs.size() == bboxes.size());
    assert(zs.size() == frames_cTw_.size());
    for (auto const& bbox : bboxes) 
    {
      // xmax, ymax must be greater than xmin, ymin
      assert((bbox.array().head<2>() < bbox.array().tail<2>()).all());
    }
}

VectorXd ObjectLM::ErrorBBoxQuadric::operator() (const LMObjectState& object, const size_t frameid) const
{
    auto bbox = bboxes_[frameid];
    auto wTo = object.getwTo().matrix();
    auto object_shape = object.getShape();
    auto Qi = ellipse_from_shape(object_shape);
    auto P = camera_intrinsics_ * (frames_cTw_[frameid] * wTo).topRows<3>();
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

int ObjectLM::ErrorBBoxQuadric::operator()(const InputType& object,
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

int ObjectLM::ErrorBBoxQuadric::df(const LMObjectState& object,
                        const size_t frameid, Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite() );

    auto wTo = object.getwTo().matrix();
    auto object_shape = object.getShape();
    auto Qi = ellipse_from_shape(object_shape);

    auto P = camera_intrinsics_ * frames_cTw_[frameid].topRows<3>();

    auto lines = poly2lineh(bbox2poly(bboxes_[frameid]));

    // jacobian wrt object state 
    fjac.setZero();
    
    for (int i = 0; i < lines.rows(); ++i) 
    {

        Eigen::Matrix<double, 1, 4> yyw = lines.row(i) * P;

        auto yyo = yyw * wTo;
        
        Eigen::Matrix<double, 1, 6> p_be_p_cxi;

        if (!use_new_bbox_residual_flag_)
        {
            // jacobian wrt old bounding box residual 
            if (use_left_perturbation_flag_)
            {
                // using left perturbation 
                Eigen::Matrix<double, 1, 6> p_eb_p_oxi = 2 * yyo * Qi * wTo.transpose() * circledCirc(yyw.transpose().block<4, 1>(0,0)).transpose();

                // jacobian wrt object pose 
                fjac.block<1, LMObjectState::wTo_DoF>(i, object.block_start_wTo()) = p_eb_p_oxi;

            }
            else 
            {

                // using right perturbation 
                // jacobian wrt object pose
                fjac.block<1, LMObjectState::wTo_DoF>(i, object.block_start_wTo())
                        = 2 * yyo * Qi * circledCirc(wTo.transpose() * yyw.transpose().block<4, 1>(0,0)).transpose();

            }

            // jacobian wrt keypoints 
            fjac.block<1, LMObjectState::ShapeDoF>(i, object.block_start_shape()) = 2 * object_shape
                            .cwiseProduct(yyo.transpose().head<3>().array().square().matrix())
                            .transpose();

        }
        else 
        {
            // jacobian wrt new bounding box residual
            Eigen::Vector3d uline_zb = lines.row(i).transpose();
            Eigen::Vector4d uline_b = P.transpose() * uline_zb;
            Eigen::Vector3d b = uline_b.block<3, 1>(0, 0);
            double b_norm = b.norm();

            Eigen::Matrix4Xd p_ulineb_p_Oxi;

            if (use_left_perturbation_flag_)
            {
                // jacobian wrt object pose
                p_ulineb_p_Oxi = wTo.transpose() * circledCirc(yyw.transpose().block<4, 1>(0,0)).transpose();
            
            }
            else 
            {
                // jacobian wrt object pose
                p_ulineb_p_Oxi = circledCirc(wTo.transpose() * yyw.transpose().block<4, 1>(0,0)).transpose();
            
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

            // jacobian wrt object pose
            fjac.block<1, LMObjectState::wTo_DoF>(i, object.block_start_wTo())
                    = p_be_p_ulinea * p_ulinea_ulineb * p_ulineb_p_Oxi;
        
            // jacobian wrt keypoints 
            fjac.block<1, LMObjectState::ShapeDoF>(i, object.block_start_shape()) = 
                ((object_shape).cwiseProduct(b.cwiseProduct(b))).transpose() / (b_norm * sqrt_bU2b);

        }

    }

    assert(fjac.allFinite());

    return 0;
}

int ObjectLM::ErrorBBoxQuadric::df(const InputType& object,
                                   Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());
    assert (fjac.rows() == nerrors());
    assert (fjac.cols() == object.dof());

    fjac.setZero();

    for (size_t frameid = 0; frameid < numFrames(); ++frameid) 
    {
        df(object, frameid, 
            fjac.block(block_start_frame(frameid), 0, ErrPerFrame, object.dof()));
    }

    assert(fjac.allFinite());
    
    return 0;
}

double
ObjectLM::ErrorBBoxQuadric::scaled_norm (const DiffType& diag, const InputType& object) const
{
    return object.scaled_norm(diag);
}


constexpr int ObjectLM::ErrorDeformRegularization::ErrPerKeypt;

ObjectLM::ErrorDeformRegularization::ErrorDeformRegularization(
        const vector_eigen<Eigen::MatrixX2d>& zs,
        const Eigen::MatrixX3d &object_keypoints_mean)
    : DenseFunctorObjectState(LMObjectState::DoF(object_keypoints_mean.rows()), NErrors(zs, object_keypoints_mean)),
      object_shape_without_deform_(object_keypoints_mean),
      valid_zs_counts_cum_(partial_sum_valid_counts(zs))
{
}

int ObjectLM::ErrorDeformRegularization::operator()(const LMObjectState& object,
                                                    const size_t frameid,
                                                    Eigen::Ref<Eigen::VectorXd> fvec) const
{
    assert(object.allFinite());
    size_t m = object.getHomSemanticKeyPts().rows();
    Eigen::Matrix3Xd diff_t =
        (object.getHomSemanticKeyPts().leftCols<3>() - object_shape_without_deform_).transpose();
    Eigen::Map<Eigen::Matrix3Xd> fvec_mat(fvec.data(), 3, m);
    fvec_mat = diff_t;
    
    // Eigen::Map<ValueType> diff_flatten(diff_t.data(), m * 3, 1);
    // fvec.noalias() = diff_flatten;
    
    assert(fvec.allFinite());
    return 0;
}

int ObjectLM::ErrorDeformRegularization::operator()(const InputType& object,
                                                    Eigen::Ref<ValueType> fvec) const
{
    int ErrPerFrame = object_shape_without_deform_.rows() * ErrPerKeypt;
    assert(object.allFinite());
    assert (fvec.rows() == static_cast<int>(ErrPerFrame*numFrames()));
    int currow = 0;
    for (size_t frameid = 0; frameid < numFrames(); ++frameid) 
    {
        this->operator ()(object, frameid, fvec.block(currow, 0, ErrPerFrame, 1));
        currow += ErrPerFrame;
    }
    assert(fvec.allFinite());
    return 0;
}

int ObjectLM::ErrorDeformRegularization::df(const LMObjectState& object,
                                            const size_t frameid,
                                            Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());
    fjac.setZero();
    for (int kpid = 0; kpid < object.getSemanticKeyPts().rows(); ++kpid) 
    {
        fjac.block(block_start_semkpt(kpid), object.block_start_sem_keypt(kpid), LMObjectState::KeyptDoF, LMObjectState::KeyptDoF)
                = Eigen::Matrix3d::Identity();
    }
    assert(fjac.allFinite());
    return 0;
}

int ObjectLM::ErrorDeformRegularization::df(const InputType& object,
                                            Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());
    assert(fjac.rows() == m_values);
    assert(fjac.cols() == m_inputs);
    fjac.setZero();
    int currow = 0;
    int ErrPerFrame = object_shape_without_deform_.rows() * ErrPerKeypt;
    for (size_t frameid = 0; frameid < numFrames(); ++frameid) 
    {
        df(object, frameid, fjac.block(currow, 0, ErrPerFrame, object.dof()));
        currow += ErrPerFrame;
    }
    assert(fjac.allFinite());
    return 0;
}


constexpr int ObjectLM::ErrorQuadVRegularization::ErrPerFrame;

ObjectLM::ErrorQuadVRegularization::ErrorQuadVRegularization(
                const vector_eigen<Eigen::MatrixX2d>& zs,
                const Eigen::MatrixX3d& object_keypoints_mean,
                const Eigen::Vector3d& object_mean_shape)
    : DenseFunctorObjectState(LMObjectState::DoF(object_keypoints_mean.rows()), NErrors(zs)),
      object_mean_shape_(object_mean_shape),
      valid_zs_counts_cum_(partial_sum_valid_counts(zs))
{
}

int ObjectLM::ErrorQuadVRegularization::operator()(const InputType& object,
                                                   Eigen::Ref<ValueType> fvec) const
{
    assert(object.allFinite());

    for (size_t frameid = 0; frameid < numFrames(); ++frameid) 
    {
        fvec.block<ErrPerFrame, 1>(frameid*ErrPerFrame, 0) = (object.getShape() - object_mean_shape_).cast<typename ValueType::Scalar>();
    }
    assert(fvec.allFinite());
    return 0;
}

int ObjectLM::ErrorQuadVRegularization::df(const InputType& object,
                                           Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());
    assert(fjac.rows() == m_values);
    assert(fjac.cols() == m_inputs);
    fjac.setZero();
    for (size_t frameid = 0; frameid < numFrames(); ++frameid) 
    {
        fjac.block<ErrPerFrame, LMObjectState::ShapeDoF>(
                    frameid * ErrPerFrame, object.block_start_shape()) = Eigen::Matrix3d::Identity();
    }
    assert(fjac.allFinite());
    return 0;
}

int ObjectLM::operator() (const InputType& object, Eigen::Ref<ValueType> fvec) const
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

int ObjectLM::df(const InputType& object, Eigen::Ref<JacobianType> fjac) const
{
    assert(object.allFinite());

    assert(fjac.rows() == block_start_functor(4));
    assert(fjac.cols() == object.dof());

    fjac.setZero();

    const std::vector<std::string> names{"feature", "bbox", "deform_reg", "quad_reg"};
    for (size_t i = 0; i < residual_functors_.size(); ++i) 
    {
        auto fjac_functor = fjac.block(block_start_functor(i), 0,
                                        residual_functors_[i]->values(), object.dof());
        residual_functors_[i]->df(object, fjac_functor);
        fjac_functor *= residual_weights_(i);
    }

    // Apply huber
    ValueType fvec(block_start_functor(4));
    this->operator()(object, fvec);
    huber_.df(fvec, fjac, fjac);

    assert(fjac.allFinite());
    
    return 0;
}

double ObjectLM::scaled_norm(const DiffType& diag, const InputType& object) const
{
    return object.scaled_norm(diag);
}

int ObjectLM::Huber::operator()(const InputType& x, Eigen::Ref<ValueType> fvec) const {
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

int ObjectLM::Huber::df(const InputType& x, const Eigen::MatrixXd& fwdJac,
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


std::vector<int> ObjectLM::get_zs_num_wrt_timestamps() const {
    return valid_zs_num_per_frame_;
}

const std::string& LevenbergMarquardtStatusString(int status)
{
    const static std::unordered_map<int, std::string> status2name{
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::NotStarted, "NotStarted"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::Running, "Running"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::ImproperInputParameters, "ImproperInputParameters"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::RelativeReductionTooSmall, "RelativeReductionTooSmall"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::RelativeErrorTooSmall, "RelativeErrorTooSmall"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall, "RelativeErrorAndReductionTooSmall"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::CosinusTooSmall, "CosinusTooSmall"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::TooManyFunctionEvaluation, "TooManyFunctionEvaluation"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::FtolTooSmall, "FtolTooSmall"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::XtolTooSmall, "XtolTooSmall"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::GtolTooSmall, "GtolTooSmall"},
        {EigenLevenbergMarquardt::LevenbergMarquardtSpace::UserAsked, "UserAsked"}};
    return status2name.at(status);
}

} // namespace orcvio
