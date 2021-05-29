#include <unordered_map>
#include <numeric>

#include "sophus/se3.hpp"

#include "orcvio/obj/ObjectLM.h"
#include "orcvio/obj/ObjectLMLite.h"
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
using Eigen::RowMajor;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using std::cout;
using std::get;
using std::make_shared;
using std::vector;

namespace Eigen
{
    template <typename Scalar, int NR = Eigen::Dynamic, int NC = 1>
    Scalar *begin(Eigen::Matrix<Scalar, NR, NC> &m)
    {
        return m.data();
    }

    template <typename Scalar, int NR = Eigen::Dynamic, int NC = 1>
    Scalar *end(Eigen::Matrix<Scalar, NR, NC> &m)
    {
        return m.data() + m.size();
    }

    template <typename Scalar, int NR = Eigen::Dynamic, int NC = 1>
    const Scalar *begin(const Eigen::Matrix<Scalar, NR, NC> &m)
    {
        return m.data();
    }

    template <typename Scalar, int NR = Eigen::Dynamic, int NC = 1>
    const Scalar *end(const Eigen::Matrix<Scalar, NR, NC> &m)
    {
        return m.data() + m.size();
    }

} // namespace Eigen

namespace orcvio
{

LMObjectStateLite operator+(const LMObjectStateLite&  x, const LMObjectStateLite::Tangent& dx)
{
    LMSE3 sum_frames_wTo(x.wTo_);
    sum_frames_wTo = x.wTo_ + dx.block<LMObjectStateLite::wTo_DoF, 1>(x.block_start_wTo(), 0);

    auto sum_shape = x.shape_ + dx.block<LMObjectStateLite::ShapeDoF, 1>(x.block_start_shape(), 0);

    // ignore keypoint deformation 
    auto sum_deformation = x.semantic_key_pts_ * 0;

    LMObjectStateLite object(sum_frames_wTo,
                         sum_shape,
                         sum_deformation);
    return object;
}

LMObjectStateLite&  LMObjectStateLite::operator+=(const LMObjectStateLite::Tangent& dx) {
    wTo_ = wTo_ + dx.block<LMObjectStateLite::wTo_DoF, 1>(block_start_wTo(), 0);

    shape_ = shape_ + dx.block<LMObjectStateLite::ShapeDoF, 1>(block_start_shape(), 0);
    // ignore keypoint deformation 
    semantic_key_pts_ = semantic_key_pts_ * 0;

    hom_semantic_key_pts_ = toHomSemanticKeyPts(semantic_key_pts_);
    return *this;
}

std::ostream& operator<<(std::ostream& o, const LMObjectStateLite&  x)
{
    o << "LMObject(wTo=";
    o << x.wTo_ << ", ";
    o << ", deformation=" << x.semantic_key_pts_ << ", shape=" << x.shape_ << ")";
    return o;
}

ObjectLMLite::ObjectLMLite(const Eigen::Vector3d &object_mean_shape,
                            const Eigen::Matrix3d &camera_intrinsics,
                            const Eigen::MatrixX3d &object_keypoints_mean,
                            const ObjectFeature &features,
                            const vector_eigen<Eigen::Matrix4d> &camposes,
                            const Eigen::VectorXd &residual_weights,
                            const bool use_left_perturbation_flag,
                            const bool use_new_bbox_residual_flag,
                            const double huber_epsilon)
    : DenseFunctorObjectStateLite(LMObjectStateLite::DoF(),
                                ErrorBBoxQuadric::NErrors(features.zb) + ErrorQuadVRegularization::NErrors(features.zb)),
        object_mean_shape_(object_mean_shape),
        camera_intrinsics_(camera_intrinsics),
        object_keypoints_mean_(object_keypoints_mean),
        features_(features),
        residual_functors_({make_shared<ErrorBBoxQuadric>(features.zs,
                                                        features.zb,
                                                        camposes,
                                                        camera_intrinsics,
                                                        object_keypoints_mean,
                                                        use_left_perturbation_flag,
                                                        use_new_bbox_residual_flag),
                            make_shared<ErrorQuadVRegularization>(features.zb,
                                                                object_keypoints_mean,
                                                                object_mean_shape)}),
        residual_weights_(residual_weights),
        huber_(huber_epsilon)
{
}

constexpr int ObjectLMLite::ErrorBBoxQuadric::ErrPerFrame;

ObjectLMLite::ErrorBBoxQuadric::ErrorBBoxQuadric(const vector_eigen<Eigen::MatrixX2d> &zs,
                                                    const vector_eigen<Eigen::Vector4d> &bboxes,
                                                    const vector_eigen<Eigen::Matrix4d> &cTw,
                                                    const Eigen::Matrix3d &camera_intrinsics,
                                                    const Eigen::MatrixX3d &object_keypoints_mean,
                                                    const bool use_left_perturbation_flag,
                                                    const bool use_new_bbox_residual_flag)
    : DenseFunctorObjectStateLite(LMObjectStateLite::DoF(), NErrors(bboxes)),
        bboxes_(bboxes),
        frames_cTw_(cTw),
        camera_intrinsics_(camera_intrinsics),
        valid_indices_(filter_valid_indices(zs)),
        use_left_perturbation_flag_(use_left_perturbation_flag),
        use_new_bbox_residual_flag_(use_new_bbox_residual_flag)
    {
        assert(zs.size() == bboxes.size());
        assert(zs.size() == frames_cTw_.size());
        for (auto const &bbox : bboxes)
        {
            // xmax, ymax must be greater than xmin, ymin
            assert((bbox.array().head<2>() < bbox.array().tail<2>()).all());
        }
    }

    VectorXd ObjectLMLite::ErrorBBoxQuadric::operator()(const LMObjectStateLite &object, const size_t frameid) const
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

    int ObjectLMLite::ErrorBBoxQuadric::operator()(const InputType &object,
                                                   Eigen::Ref<ValueType> fvec) const
    {
        assert(object.allFinite());
        int N = bboxes_.size();
        assert(fvec.rows() == ErrPerFrame * N);
        int currow = 0;
        for (size_t frameid = 0; frameid < bboxes_.size(); ++frameid)
        {
            auto f = this->operator()(object, frameid);
            fvec.block(currow, 0, f.rows(), 1) = f;
            currow += f.rows();
        }
        assert(fvec.allFinite());
        return 0;
    }

    int ObjectLMLite::ErrorBBoxQuadric::df(const LMObjectStateLite &object,
                                           const size_t frameid, Eigen::Ref<JacobianType> fjac) const
    {
        assert(object.allFinite());

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
                    Eigen::Matrix<double, 1, 6> p_eb_p_oxi = 2 * yyo * Qi * wTo.transpose() * circledCirc(yyw.transpose().block<4, 1>(0, 0)).transpose();

                    // jacobian wrt object pose
                    fjac.block<1, LMObjectStateLite::wTo_DoF>(i, object.block_start_wTo()) = p_eb_p_oxi;
                }
                else
                {

                    // using right perturbation
                    // jacobian wrt object pose
                    fjac.block<1, LMObjectStateLite::wTo_DoF>(i, object.block_start_wTo()) = 2 * yyo * Qi * circledCirc(wTo.transpose() * yyw.transpose().block<4, 1>(0, 0)).transpose();
                }

                // jacobian wrt keypoints
                fjac.block<1, LMObjectStateLite::ShapeDoF>(i, object.block_start_shape()) = 2 * object_shape
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
                    p_ulineb_p_Oxi = wTo.transpose() * circledCirc(yyw.transpose().block<4, 1>(0, 0)).transpose();
                }
                else
                {
                    // jacobian wrt object pose
                    p_ulineb_p_Oxi = circledCirc(wTo.transpose() * yyw.transpose().block<4, 1>(0, 0)).transpose();
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
                fjac.block<1, LMObjectStateLite::wTo_DoF>(i, object.block_start_wTo()) = p_be_p_ulinea * p_ulinea_ulineb * p_ulineb_p_Oxi;

                // jacobian wrt keypoints
                fjac.block<1, LMObjectStateLite::ShapeDoF>(i, object.block_start_shape()) =
                    ((object_shape).cwiseProduct(b.cwiseProduct(b))).transpose() / (b_norm * sqrt_bU2b);
            }
        }

        assert(fjac.allFinite());

        return 0;
    }

    int ObjectLMLite::ErrorBBoxQuadric::df(const InputType &object,
                                           Eigen::Ref<JacobianType> fjac) const
    {
        assert(object.allFinite());
        assert(fjac.rows() == nerrors());
        assert(fjac.cols() == object.dof());

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
    ObjectLMLite::ErrorBBoxQuadric::scaled_norm(const DiffType &diag, const InputType &object) const
    {
        return object.scaled_norm(diag);
    }

    constexpr int ObjectLMLite::ErrorQuadVRegularization::ErrPerFrame;

    ObjectLMLite::ErrorQuadVRegularization::ErrorQuadVRegularization(
        const vector_eigen<Eigen::Vector4d> &zb,
        const Eigen::MatrixX3d &object_keypoints_mean,
        const Eigen::Vector3d &object_mean_shape)
        : DenseFunctorObjectStateLite(LMObjectStateLite::DoF(), NErrors(zb)),
          object_mean_shape_(object_mean_shape),
          zb_(zb)
    {
    }

    int ObjectLMLite::ErrorQuadVRegularization::operator()(const InputType &object,
                                                           Eigen::Ref<ValueType> fvec) const
    {
        assert(object.allFinite());

        for (size_t frameid = 0; frameid < numFrames(); ++frameid)
        {
            fvec.block<ErrPerFrame, 1>(frameid * ErrPerFrame, 0) = (object.getShape() - object_mean_shape_).cast<typename ValueType::Scalar>();
        }

        assert(fvec.allFinite());

        return 0;
    }

    int ObjectLMLite::ErrorQuadVRegularization::df(const InputType &object,
                                                   Eigen::Ref<JacobianType> fjac) const
    {
        assert(object.allFinite());
        assert(fjac.rows() == m_values);
        assert(fjac.cols() == m_inputs);
        fjac.setZero();
        for (size_t frameid = 0; frameid < numFrames(); ++frameid)
        {
            fjac.block<ErrPerFrame, LMObjectStateLite::ShapeDoF>(
                frameid * ErrPerFrame, object.block_start_shape()) = Eigen::Matrix3d::Identity();
        }
        assert(fjac.allFinite());
        return 0;
    }

    int ObjectLMLite::operator()(const InputType &object, Eigen::Ref<ValueType> fvec) const
    {
        assert(object.allFinite());
        assert(fvec.rows() == block_start_functor(residual_functors_.size()));

        fvec.setZero();

        const std::vector<std::string> names{"bbox", "quad_reg"};
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

    int ObjectLMLite::df(const InputType &object, Eigen::Ref<JacobianType> fjac) const
    {
        assert(object.allFinite());

        assert(fjac.rows() == block_start_functor(2));
        assert(fjac.cols() == object.dof());

        fjac.setZero();

        const std::vector<std::string> names{"bbox", "quad_reg"};
        for (size_t i = 0; i < residual_functors_.size(); ++i)
        {
            auto fjac_functor = fjac.block(block_start_functor(i), 0,
                                           residual_functors_[i]->values(), object.dof());
            residual_functors_[i]->df(object, fjac_functor);

            if (DEBUG)
                std::cout << "jac_" << names[i] << "(jacobian):" << fjac_functor << "\n";

            fjac_functor *= residual_weights_(i);
        }

        // Apply huber
        ValueType fvec(block_start_functor(2));
        this->operator()(object, fvec);
        huber_.df(fvec, fjac, fjac);

        assert(fjac.allFinite());

        return 0;
    }

    double ObjectLMLite::scaled_norm(const DiffType &diag, const InputType &object) const
    {
        return object.scaled_norm(diag);
    }

    int ObjectLMLite::Huber::operator()(const InputType &x, Eigen::Ref<ValueType> fvec) const
    {
        if (std::isinf(huber_epsilon_))
        {
            fvec = x;
            return 0;
        }

        double k = huber_epsilon_;
        double ksq = k * k;
        // NOTE: Be careful x and fvec can be same
        for (int r = 0; r < x.rows(); ++r)
        {
            if (x(r) < ksq)
                fvec(r) = x(r);
            else
                fvec(r) = 2 * k * std::sqrt(x(r)) - ksq;
        }
        return 0;
    }

    int ObjectLMLite::Huber::df(const InputType &x, const Eigen::MatrixXd &fwdJac,
                                Eigen::Ref<JacobianType> fjac) const
    {
        if (std::isinf(huber_epsilon_))
        {
            fjac = fwdJac;
            return 0;
        }

        double k = huber_epsilon_;
        double ksq = k * k;
        // NOTE: Be careful fwdJac and fjac can be same
        for (int r = 0; r < fwdJac.rows(); ++r)
        {
            if (x(r) < ksq)
                fjac.row(r) = fwdJac.row(r);
            else
                fjac.row(r) = k / std::sqrt(x(r)) * fwdJac.row(r);
        }
        return 0;
    }

} // namespace orcvio
