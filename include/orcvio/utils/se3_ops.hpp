#ifndef SE3_OPS_HPP
#define SE3_OPS_HPP

#include <tuple>
#include <vector>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>
#include <opencv2/core/eigen.hpp>

#include "math_utils.hpp"

namespace orcvio
{

template<typename T>
  using vector_eigen = std::vector<T, Eigen::aligned_allocator<T>>;

/**
 * @brief converts vector to skew symmetric matrix in batch
 *
 * @param a: size n x 3, input vector
 *
 * @return : size n x 3 x 3, skew symmetric matrix
 */
template<typename Scalar>
vector_eigen<Eigen::Matrix<Scalar, 3, 3> > skew(const vector_eigen<Eigen::Matrix<Scalar, 3, 1> > &a)
{

    vector_eigen<Eigen::Matrix<Scalar, 3, 3> > S;

    for (const auto& w : a)
    {
        Eigen::Matrix<Scalar, 3, 3> w_x = skewSymmetric(w);
        S.push_back(w_x);
    }

    return S;

}

/**
 * @brief converts 6-vector to 4x4 hat form in se(3) in batch
 *
 * @param x: size n x 6, n se3 elements
 *
 * @return : size n x 4 x 4, n elements of se(3)
 */
template<typename Scalar>
vector_eigen<Eigen::Matrix<Scalar, 4, 4> > axangle2twist(const vector_eigen<Eigen::Matrix<Scalar, 6, 1> > &a)
{

    vector_eigen<Eigen::Matrix<Scalar, 4, 4> > T;

    for (auto v : a)
    {
        Eigen::Matrix<Scalar, 4, 4> v_x = Eigen::Matrix4d::Zero();
        v_x(0, 1) = -v(5);
        v_x(0, 2) = v(4);
        v_x(0, 3) = v(0);
        v_x(1, 0) = v(5);
        v_x(1, 2) = -v(3);
        v_x(1, 3) = v(1);
        v_x(2, 0) = -v(4);
        v_x(2, 1) = v(3);
        v_x(2, 3) = v(2);

        T.push_back(v_x);
    }

    return T;

}

/**
 * @brief converts se3 element to SE3 in batch
 *
 * @param x: size n x 6, n se3 elements
 *
 * @return : size n x 4 x 4, n elements of SE(3)
 */
template<typename Scalar>
vector_eigen<Eigen::Matrix<Scalar, 4, 4> > axangle2pose(const vector_eigen<Eigen::Matrix<Scalar, 6, 1> > &a)
{

    vector_eigen<Eigen::Matrix<Scalar, 4, 4> > T;

    for (auto x : a)
    {

        Sophus::SE3d T_temp = Sophus::SE3d::exp(x);
        T.push_back(T_temp.matrix());

    }

    return T;

}

/**
 * @brief converts axis angle to SO3 in batch
 *
 * @param a = n x 3 = n axis-angle elements
 *
 * @return : R = n x 3 x 3 = n elements of SO(3)
 */
template<typename Scalar>
vector_eigen<Eigen::Matrix<Scalar, 3, 3> > axangle2rot(const vector_eigen<Eigen::Matrix<Scalar, 3, 1> > &a)
{

    vector_eigen<Eigen::Matrix<Scalar, 3, 3> > R;

    for (auto x : a)
    {

        Sophus::SO3d R_temp = Sophus::SO3d::exp(x);
        R.push_back(R_temp.matrix());

    }

    return R;

}


/**
 * @brief performs batch inverse of transform matrix
 *
 * @param T: size n x 4 x 4, n elements of SE(3)
 *
 * @return : size n x 4 x 4, inverse of T
 */
template<typename Scalar>
vector_eigen<Eigen::Matrix<Scalar, 4, 4> > inversePose(const vector_eigen<Eigen::Matrix<Scalar, 4, 4> > &T)
{

    vector_eigen<Eigen::Matrix<Scalar, 4, 4> > iT;

    for (auto T_temp : T)
    {
        Eigen::Matrix<Scalar, 4, 4> iT_temp = Eigen::Matrix4d::Zero();

        iT_temp(0, 0) = T_temp(0, 0);
        iT_temp(0, 1) = T_temp(1, 0);
        iT_temp(0, 2) = T_temp(2, 0);

        iT_temp(1, 0) = T_temp(0, 1);
        iT_temp(1, 1) = T_temp(1, 1);
        iT_temp(1, 2) = T_temp(2, 1);

        iT_temp(2, 0) = T_temp(0, 2);
        iT_temp(2, 1) = T_temp(1, 2);
        iT_temp(2, 2) = T_temp(2, 2);

        iT_temp.block(0, 3, 3, 1) = -1 * iT_temp.block(0, 0, 3, 3) * T_temp.block(0, 3, 3, 1);

        iT_temp.block(3, 0, 1, 4) = T_temp.block(3, 0, 1, 4);

        iT.push_back(iT_temp);

    }

    return iT;

}


/**
 * @brief odot operator
 *
 * \f{align*}{
 *  \underline{x}^{\odot} = \begin{bmatrix} I_{3\times 3} & -x_{\times} \\ 0 & 0\end{bmatrix}
 @f}
 *
 * @param ph = 4 = point in homogeneous coordinates
 *
 * @return : odot(ph) = 4 x 6
 */
template<typename Derived>
 Eigen::Matrix<typename Derived::Scalar, 4, 6> odotOperator(const Eigen::MatrixBase<Derived>& x) {
     assert(x.rows() == 4);
     assert(x.cols() == 1);

   Eigen::Matrix<typename Derived::Scalar, 4, 6> temp;
   temp.setZero();

   temp.block(0, 3, 3, 3) = -1 * skewSymmetric(x.template head<3>());

   temp(0, 0) = x(3);
   temp(1, 1) = x(3);
   temp(2, 2) = x(3);
   return temp;
 }

/**
 * @brief odot operator
 *
 * \f{align*}{
 *  \underline{x}^{\odot} = \begin{bmatrix} I_{3\times 3} & -x_{\times} \\ 0 & 0\end{bmatrix}
  @f}
 *
 * @param ph = n x 4 = points in homogeneous coordinates
 *
 * @return : odot(ph) = n x 4 x 6
 */
 template<typename Scalar>
vector_eigen<Eigen::Matrix<Scalar, 4, 6> > odotOperator(const vector_eigen<Eigen::Matrix<Scalar, 4, 1> > &ph)
{

    vector_eigen<Eigen::Matrix<Scalar, 4, 6> > zz;

    for (const auto& x : ph)
    {
      zz.emplace_back(odotOperator(x));
    }
    return zz;
}

/**
  * @brief circle dot operator
  *
  * @param ph = 4 = points in homogeneous coordinates
  *
  * @return : circledCirc(ph) = 6 x 4
  */
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 6, 4> circledCirc(const Eigen::MatrixBase<Derived>& x) {
    static_assert(Derived::RowsAtCompileTime == 4, "x is not 4D vector");
    static_assert(Derived::ColsAtCompileTime == 1, "x is not a vector");
    Eigen::Matrix<typename Derived::Scalar, 6, 4> temp;
    temp.setZero();

    temp.block(3, 0, 3, 3) = -1 * skewSymmetric(x.template block<3, 1>(0, 0));

    temp.block(0, 3, 3, 1) = x.template block<3, 1>(0, 0);
    return temp;
}

/**
 * @brief circle dot operator
 *
 * @param ph = n x 4 = points in homogeneous coordinates
 *
 * @return : circledCirc(ph) = n x 6 x 4
 */
template<typename Scalar>
vector_eigen<Eigen::Matrix<Scalar, 6, 4> > circledCirc(const vector_eigen<Eigen::Matrix<Scalar, 4, 1> > &ph)
{

    vector_eigen<Eigen::Matrix<Scalar, 6, 4>  > zz;

    for (const auto& x : ph)
    {
      zz.push_back(circledCirc(x));
    }

    return zz;

}

/**
 * @brief only keep yaw and zero z
 *
 * @param T_SE3: 4 x 4
 *
 * @return T_SE2: 4 x 4
 */

template<typename Scalar>
Eigen::Matrix<Scalar, 4, 4> poseSE32SE2(const Eigen::Matrix<Scalar, 4, 4> &T_SE3)
{

    Eigen::Matrix<Scalar, 4, 4> T_SE2 = Eigen::Matrix4d::Identity();

    // yaw: alpha=arctan(r21/r11)
    Scalar yaw = M_PI/atan2(T_SE3(1, 0), T_SE3(0, 0));

    // deal with the case when yaw is nan  
    if (!std::isfinite(yaw))
      yaw = 0; 

    T_SE2(0, 0) = cos(yaw);
    T_SE2(0, 1) = -sin(yaw);
    T_SE2(0, 3) = T_SE3(0, 3);

    T_SE2(1, 0) = sin(yaw);
    T_SE2(1, 1) = cos(yaw);
    T_SE2(1, 3) = T_SE3(1, 3);

    return T_SE2;

}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 2, Eigen::Dynamic>
project_image(const Eigen::MatrixBase<Derived>& uv_hom) {
  auto den = uv_hom.template bottomRows<1>().array().template replicate<2, 1>();
  auto num = uv_hom.template topRows<2>().array();
  return ( num / den ).matrix();
}

template<typename Scalar>
using Matrix23 = Eigen::Matrix<Scalar, 2, 3>;

/**
 * @brief Differentiate \f$ \pi([x, y, z]^\top) = [x/z, y/z] \f$
 *
 * \f{align*}{
 * \frac{\partial \pi([x, y, z]^\top)}{\partial \mathbf{x}}
 * = [1/z, 0,   -x/z^2]
 *   [0,   1/z, -y/z^2]
 * @f}

 * @param x
 * @return Jacobian
 */
template <typename Derived>
Matrix23<typename Derived::Scalar>
  project_image_df(const Eigen::MatrixBase<Derived>& x)
{
    static_assert(Derived::RowsAtCompileTime == 3, " need 3 x 1 vector");
    static_assert(Derived::ColsAtCompileTime == 1, " need 3 x 1 vector");
    typedef typename Derived::Scalar Scalar;
   Scalar z = x(2, 0);
  Scalar zsq = z * z;
  Matrix23<typename Derived::Scalar> df;
  df <<
    1/z,   0, -x(0, 0)/zsq,
    0  , 1/z, -x(1, 0)/zsq;
  return df;
}

/**
 * @brief project_object_points
 *
 * @param P   : Camera projection matrix (3 x 4)
 * @param wTo : Object to world transform (4 x 4)
 * @param points_w : Points (n x 4), note this is points in object frame with homo coord 
 * @return Points (n x 2)
 */
template <typename D1, typename D2, typename D3>
Eigen::Matrix<typename D3::Scalar, Eigen::Dynamic, 2>
project_object_points(const Eigen::MatrixBase<D1>& P,
                   const Eigen::MatrixBase<D2>& wTo, const Eigen::MatrixBase<D3>& points_w) {
  auto uv_hom = P * (wTo * points_w.transpose());
  return project_image(uv_hom).transpose();
}


/**
 * @brief Computes the derivative of projection operation wrt object pose
 *
 * \f[
 * \frac{\partial \pi(K ^CT_w(\chi) x_w)}{\partial \xi} = \frac{\partial \pi(K T x_w)}{\partial T} @ \frac{^cT_w(\chi)}{\partial \xi}
 * \f]
 *
 * @param [in] P   : Camera projection matrix (3 x 4)
 * @param [in] wTo : World transform (4 x 4)
 * @param [in] points_o : Points in object frame (n x 4)
 *
 * @return Jacobians (n x 2 x 6)
 */
template<typename D1, typename D2, typename D3>
Eigen::Matrix<typename D3::Scalar, Eigen::Dynamic, 6>
  project_object_points_df_object(const Eigen::MatrixBase<D1>& P, const Eigen::MatrixBase<D2>& wTo,
                           const Eigen::MatrixBase<D3>& points_o,
                           const bool use_left_perturbation_flag) {
  auto X_o = points_o.transpose();
  Eigen::Matrix<typename D3::Scalar, Eigen::Dynamic, 6> jacobians(2*points_o.rows(), 6);
  for (int i = 0; i < points_o.rows(); ++i) {

    auto dpibydx = project_image_df(P * wTo * X_o.col(i).template topLeftCorner<4, 1>());
    
    Eigen::MatrixXd jac;
    if (use_left_perturbation_flag)
    {
      // using left perturbation 
      jac = dpibydx * P * odotOperator(wTo * X_o.col(i));
    }
    else 
    {
      // using right perturbation 
      jac = dpibydx * P * wTo * odotOperator(X_o.col(i));
    }

    assert(jac.rows() == 2 && jac.cols() == 6);
    jacobians.template block<2, 6>(2*i, 0) = jac.template block<2, 6>(0,0);
    
  }

  return jacobians;
}

/**
 * @brief Computes the derivative of projection operation wrt camera pose
 * @param [in] P   : Camera projection matrix (3 x 4)
 * @param [in] wTo : Object frame to World frame transformation (4 x 4)
 * @param [in] cTw : world frame to camera frame transformation (4 x 4)
 * @param [in] points_o : Points in object frame (n x 4)
 *
 * @return Jacobians ((nk x 2) x 6)
 */
template<typename D1, typename D2, typename D3>
Eigen::Matrix<typename D3::Scalar, Eigen::Dynamic, 6>
  project_object_points_df_camera(const Eigen::MatrixBase<D1>& P, 
                           const Eigen::MatrixBase<D2>& wTo,
                           const Eigen::MatrixBase<D2>& cTw,
                           const Eigen::MatrixBase<D3>& points_o,
                           const bool use_left_perturbation_flag) {
  
  auto X_o = points_o.transpose();
  Eigen::Matrix<typename D3::Scalar, Eigen::Dynamic, 6> jacobians(2*points_o.rows(), 6);

  Eigen::MatrixXd ps_puline_s = Eigen::Matrix<double, 3, 4>::Zero();
  ps_puline_s.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

  for (int i = 0; i < points_o.rows(); ++i) {

    auto dpibydx = project_image_df(P * wTo * X_o.col(i).template topLeftCorner<4, 1>());

    Eigen::MatrixXd jac;
    if (use_left_perturbation_flag)
    {
      // using left perturbation 
      jac = -1 * dpibydx * ps_puline_s * cTw * odotOperator(wTo * X_o.col(i));
      // or equivalently 
      // jac = -1 * dpibydx * P * odotOperator(wTo * X_o.col(i));
      // std::cerr << "jac equivalent " << odotOperator(wTo * X_o.col(i)) << "\n";
    }
    else 
    {
      // using right perturbation 
      jac = -1 * dpibydx * ps_puline_s * odotOperator(cTw * wTo * X_o.col(i));
    }

    assert(jac.rows() == 2 && jac.cols() == 6);
    jacobians.template block<2, 6>(2*i, 0) = jac.template block<2, 6>(0,0);

    // for debugging 
    // std::cerr << "jac " << jac << "\n";

  }

  return jacobians;
}

/**
 * @brief Read eigen matrices from hdfio object
 *
 * @param h5io
 * @param name
 * @return
 */
template<typename T = cv::Ptr<cv::hdf::HDF5>>
Eigen::MatrixXd
dsread(const T& h5io, const std::string& name) {
  cv::Mat m;
  if (! h5io->hlexists(name))
    throw std::runtime_error("Unable to find dataset " + name);
  h5io->dsread( m, name );
  if (m.dims > 2)
    throw std::runtime_error("Cannot handle more than 2 dims, found " + std::to_string(m.dims));
  Eigen::MatrixXd m_e;
  cv::cv2eigen(m, m_e);
  return m_e;
}


/**
 * @brief Computes the distance between two SE3 transforms
 *
 * @param T1: size 4 x 4, input transform
 * @param T2: size 4 x 4, input transform
 *
 * @return : (3-tr(R))/2, |t₁ - t₂|₂
 */
template <typename D1, typename D2>
std::tuple<typename D1::Scalar, typename D1::Scalar>
displacement(const Eigen::MatrixBase<D1>& T1, const Eigen::MatrixBase<D2>& T2)
{
  using Scalar = typename D1::Scalar;
  // tr(R) = 1 + 2 cos θ
  // 1 - cos θ = 3/2 - tr(R)/2 ∈ [0, 2]
  auto R1 = T1.template block<3,3>(0,0);
  auto R2 = T2.template block<3,3>(0,0);
  Scalar dispR = (3 - (R1.transpose() * R2).trace()) / 2;
  Scalar dispt = (T1.template topRightCorner<3,1>() - T2.template topRightCorner<3,1>()).norm();
  return std::make_tuple(dispR, dispt);
}

/**
 * @brief odot operator
 *
 * \f{align*}{
 *  \underline{x}^{\odot} = \begin{bmatrix} I_{3\times 3} & -x_{\times} \\ 0 & 0\end{bmatrix}
 @f}
 *
 * @param ph = 4 = point in homogeneous coordinates
 *
 * @return : odot(ph) = 4 x 6
 */
inline Eigen::Matrix<double, 4, 6> odotOperator(const Eigen::Vector4d& x) {
  Eigen::Matrix<double, 4, 6> temp;
  temp.setZero();
  temp.block(0, 3, 3, 3) = -1 * skewSymmetric(x.head(3));

  temp(0, 0) = x(3);
  temp(1, 1) = x(3);
  temp(2, 2) = x(3);
  return temp;
}

/**
 * @brief Computes the derivative of camera se3 wrt IMU se3 
 * @param [in] R_b2c : rotation of body frame to camera frame 
 * @param [in] t_c_b : position of camera frame in body frame 
 * @param [in] R_w2c : rotation of world frame to camera frame 
 * @param [in] t_b_w : position of body frame in world frame 
 * @param [in] use_left_perturbation_flag : which perturbation to use 
 *
 * @return Jacobians (6 x 6)
 */
inline Eigen::Matrix<double, 6, 6> get_cam_wrt_imu_se3_jacobian(const Eigen::Matrix3d& R_b2c, const Eigen::Vector3d& t_c_b, const Eigen::Matrix3d& R_w2c, const Eigen::Vector3d& t_b_w, const bool use_left_perturbation_flag)
{
  Eigen::Matrix<double, 6, 6> p_cxi_p_ixi = Eigen::Matrix<double, 6, 6>::Zero();
  if (use_left_perturbation_flag)
  {

    p_cxi_p_ixi.block<3, 3>(0, 0) = skewSymmetric(t_b_w);
    p_cxi_p_ixi.block<3, 3>(3, 0) = Eigen::Matrix<double, 3, 3>::Identity();
    p_cxi_p_ixi.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();

  }
  else 
  {

    p_cxi_p_ixi.block<3, 3>(0, 0) = -1 * R_b2c * skewSymmetric(t_c_b);
    p_cxi_p_ixi.block<3, 3>(3, 0) = R_b2c;
    p_cxi_p_ixi.block<3, 3>(0, 3) = R_w2c;

  }

  return p_cxi_p_ixi;
}

} // namespace orcvio
#endif // SE3_OPS_HPP
