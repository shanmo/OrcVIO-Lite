/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Some changes have been made to use it in orcvio

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

namespace orcvio {

/*
 *  @brief Create a skew-symmetric matrix from a 3-element vector.
 *  @note Performs the operation:
 *  w   ->  [  0 -w3  w2]
 *          [ w3   0 -w1]
 *          [-w2  w1   0]
 */
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w) {
  Eigen::Matrix3d w_hat;
  w_hat(0, 0) = 0;
  w_hat(0, 1) = -w(2);
  w_hat(0, 2) = w(1);
  w_hat(1, 0) = w(2);
  w_hat(1, 1) = 0;
  w_hat(1, 2) = -w(0);
  w_hat(2, 0) = -w(1);
  w_hat(2, 1) = w(0);
  w_hat(2, 2) = 0;
  return w_hat;
}
//     /**
//      * @brief Skew-symmetric matrix from a given 3x1 vector
//      *
//      * This is based on equation 6 in [Indirect Kalman Filter for 3D Attitude Estimation](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf):
//      * \f{align*}{
//      *  \lfloor\mathbf{v}\times\rfloor =
//      *  \begin{bmatrix}
//      *  0 & -v_3 & v_2 \\ v_3 & 0 & -v_1 \\ -v_2 & v_1 & 0
//      *  \end{bmatrix}
//      * @f}
//      *
//      * @param[in] w 3x1 vector to be made a skew-symmetric
//      * @return 3x3 skew-symmetric matrix
//      */
// template<typename Der>
//     Eigen::Matrix<typename Der::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Der> &w) {
//         static_assert(Der::RowsAtCompileTime == 3, "3 vector");
//         static_assert(Der::ColsAtCompileTime == 1, "a vector");
//         Eigen::Matrix<typename Der::Scalar, 3, 3> w_x;
//         w_x << 0, -w(2, 0), w(1,0),
//                 w(2, 0), 0, -w(0, 0),
//                 -w(1, 0), w(0, 0), 0;
//         return w_x;
//     }

/*
 * @brief Normalize the given quaternion to unit quaternion.
 */
inline void quaternionNormalize(Eigen::Vector4d& q) {
  double norm = q.norm();
  q = q / norm;
  return;
}

/*
 * @brief Perform q1 * q2.
 *  
 *    Format of q1 and q2 is as [x,y,z,w]
 */
inline Eigen::Vector4d quaternionMultiplication(
    const Eigen::Vector4d& q1,
    const Eigen::Vector4d& q2) {
  Eigen::Matrix4d L;

  // Hamilton
  L(0, 0) =  q1(3); L(0, 1) = -q1(2); L(0, 2) =  q1(1); L(0, 3) =  q1(0);
  L(1, 0) =  q1(2); L(1, 1) =  q1(3); L(1, 2) = -q1(0); L(1, 3) =  q1(1);
  L(2, 0) = -q1(1); L(2, 1) =  q1(0); L(2, 2) =  q1(3); L(2, 3) =  q1(2);
  L(3, 0) = -q1(0); L(3, 1) = -q1(1); L(3, 2) = -q1(2); L(3, 3) =  q1(3);

  Eigen::Vector4d q = L * q2;
  quaternionNormalize(q);
  return q;
}

/*
 * @brief Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
inline Eigen::Vector4d smallAngleQuaternion(
    const Eigen::Vector3d& dtheta) {

  Eigen::Vector3d dq = dtheta / 2.0;
  Eigen::Vector4d q;
  double dq_square_norm = dq.squaredNorm();

  if (dq_square_norm <= 1) {
    q.head<3>() = dq;
    q(3) = std::sqrt(1-dq_square_norm);
  } else {
    q.head<3>() = dq;
    q(3) = 1;
    q = q / std::sqrt(1+dq_square_norm);
  }

  return q;
}

/*
 * @brief Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
inline Eigen::Quaterniond getSmallAngleQuaternion(
    const Eigen::Vector3d& dtheta) {

  Eigen::Vector3d dq = dtheta / 2.0;
  Eigen::Quaterniond q;
  double dq_square_norm = dq.squaredNorm();

  if (dq_square_norm <= 1) {
    q.x() = dq(0);
    q.y() = dq(1);
    q.z() = dq(2);
    q.w() = std::sqrt(1-dq_square_norm);
  } else {
    q.x() = dq(0);
    q.y() = dq(1);
    q.z() = dq(2);
    q.w() = 1;
    q.normalize();
  }

  return q;
}

/*
 * @brief Convert a quaternion to the corresponding rotation matrix
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Matrix3d quaternionToRotation(
    const Eigen::Vector4d& q) {
  // Hamilton
  const double& qw = q(3);
  const double& qx = q(0);
  const double& qy = q(1);
  const double& qz = q(2);
  Eigen::Matrix3d R;
  R(0, 0) = 1-2*(qy*qy+qz*qz);  R(0, 1) =   2*(qx*qy-qw*qz);  R(0, 2) =   2*(qx*qz+qw*qy);
  R(1, 0) =   2*(qx*qy+qw*qz);  R(1, 1) = 1-2*(qx*qx+qz*qz);  R(1, 2) =   2*(qy*qz-qw*qx);
  R(2, 0) =   2*(qx*qz-qw*qy);  R(2, 1) =   2*(qy*qz+qw*qx);  R(2, 2) = 1-2*(qx*qx+qy*qy);

  return R;
}

/*
 * @brief Convert a rotation matrix to a quaternion.
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Vector4d rotationToQuaternion(
    const Eigen::Matrix3d& R) {
  Eigen::Vector4d score;
  score(0) = R(0, 0);
  score(1) = R(1, 1);
  score(2) = R(2, 2);
  score(3) = R.trace();

  int max_row = 0, max_col = 0;
  score.maxCoeff(&max_row, &max_col);

  Eigen::Vector4d q = Eigen::Vector4d::Zero();

  // Hamilton
  if (max_row == 0) {
    q(0) = std::sqrt(1+2*R(0, 0)-R.trace()) / 2.0;
    q(1) = (R(0, 1)+R(1, 0)) / (4*q(0));
    q(2) = (R(0, 2)+R(2, 0)) / (4*q(0));
    q(3) = (R(2, 1)-R(1, 2)) / (4*q(0));
  } else if (max_row == 1) {
    q(1) = std::sqrt(1+2*R(1, 1)-R.trace()) / 2.0;
    q(0) = (R(0, 1)+R(1, 0)) / (4*q(1));
    q(2) = (R(1, 2)+R(2, 1)) / (4*q(1));
    q(3) = (R(0, 2)-R(2, 0)) / (4*q(1));
  } else if (max_row == 2) {
    q(2) = std::sqrt(1+2*R(2, 2)-R.trace()) / 2.0;
    q(0) = (R(0, 2)+R(2, 0)) / (4*q(2));
    q(1) = (R(1, 2)+R(2, 1)) / (4*q(2));
    q(3) = (R(1, 0)-R(0, 1)) / (4*q(2));
  } else {
    q(3) = std::sqrt(1+R.trace()) / 2.0;
    q(0) = (R(2, 1)-R(1, 2)) / (4*q(3));
    q(1) = (R(0, 2)-R(2, 0)) / (4*q(3));
    q(2) = (R(1, 0)-R(0, 1)) / (4*q(3));
  }

  if (q(3) < 0) q = -q;
  quaternionNormalize(q);
  return q;
}

inline Eigen::Matrix3d Hl_operator(const Eigen::Vector3d& gyro) {

    double gyro_norm = gyro.norm();

    Eigen::Matrix3d term1 = 0.5 * Eigen::Matrix3d::Identity();

    // handle the case when the input is close to 0 
    if (gyro_norm < 1.0e-5)
    {
        return term1;
    }

    Eigen::Matrix3d term2 = ((gyro_norm - sin(gyro_norm)) / pow(gyro_norm, 3)) * skewSymmetric(gyro);
    Eigen::Matrix3d term3 = ((2*(cos(gyro_norm) - 1) + pow(gyro_norm, 2)) / (2*pow(gyro_norm, 4))) * (skewSymmetric(gyro) * skewSymmetric(gyro));

    Eigen::Matrix3d Hl = term1 + term2 + term3; 

    return Hl;

}

inline Eigen::Matrix3d Jl_operator(const Eigen::Vector3d& gyro) {

    double gyro_norm = gyro.norm();

    Eigen::Matrix3d term1 = Eigen::Matrix3d::Identity();

    // handle the case when the input is close to 0 
    if (gyro_norm < 1.0e-5)
    {
        return term1;
    }

    Eigen::Matrix3d term2 = ((1 - cos(gyro_norm)) / pow(gyro_norm, 2)) * skewSymmetric(gyro);
    Eigen::Matrix3d term3 = ((gyro_norm - sin(gyro_norm)) / pow(gyro_norm, 3)) * skewSymmetric(gyro) * skewSymmetric(gyro);

    Eigen::Matrix3d Jl = term1 + term2 + term3; 

    return Jl;
  
}

/*
 * @brief Inverse a Hamilton quaternion 
 */
inline Eigen::Vector4d inverseQuaternion(
    const Eigen::Vector4d& quat) {

  Eigen::Vector4d q_inv;

  Eigen::Quaterniond quat_Q = Eigen::Quaterniond(quat);
  q_inv = quat_Q.inverse().coeffs();

  return q_inv;
}

inline bool nullspace_project_inplace_svd(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) 
{

  bool nullspace_trick_success_flag = true;

  if (H_f.rows() <= H_f.cols())
  {
    // insufficient residuals 
    nullspace_trick_success_flag = false; 
  }
  else 
  {

    // Project the residual and Jacobians onto the nullspace of H_f.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_helper(H_f, Eigen::ComputeFullU | Eigen::ComputeThinV);
    Eigen::MatrixXd A = svd_helper.matrixU().rightCols(
            H_f.rows() - H_f.cols());

    H_x = (A.transpose() * H_x).eval(); 
    res = (A.transpose() * res).eval();

  }

  return nullspace_trick_success_flag; 

}

// ref https://github.com/rpng/open_vins/blob/e169a77906941d662597bc83c8552e28b9a404f4/ov_msckf/src/update/UpdaterHelper.cpp
inline bool nullspace_project_inplace_qr(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

    bool nullspace_trick_success_flag = true; 

    if (H_f.rows()-H_f.cols() <= 0)
    {
      // insufficient residuals 
      nullspace_trick_success_flag = false;
    }
    else 
    {

      // ref https://docs.openvins.com/update-null.html
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(H_f.rows(), H_f.cols());
      qr.compute(H_f);
      Eigen::MatrixXd Q = qr.householderQ();
      Eigen::MatrixXd Q1 = Q.block(0,0,H_f.rows(),H_f.cols());
      Eigen::MatrixXd Q2 = Q.block(0,H_f.cols(),H_f.rows(),H_f.rows()-H_f.cols());

      H_x = (Q2.transpose() * H_x).eval();
      res = (Q2.transpose() * res).eval();

      // Sanity check
      assert(H_x.rows()==res.rows());

    }

    return nullspace_trick_success_flag;

}

inline Eigen::Vector4d unnormalize_bbox(const Eigen::Vector4d& bbox, const Eigen::Matrix3d& K) {
  Eigen::Vector4d nbbox;
  nbbox.head<2>() = K.block<2,2>(0,0) * bbox.head<2>() + K.topRightCorner<2,1>();
  nbbox.tail<2>() = K.block<2,2>(0,0) * bbox.tail<2>() + K.topRightCorner<2,1>();
  return nbbox;
}

inline Eigen::Vector4d normalize_bbox(const Eigen::Vector4d& bbox, const Eigen::Matrix3d& K) {

    // bbox format is 
    // bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

    Eigen::Vector4d nbbox;

    // for x 
    nbbox(0) = (bbox(0) - K(0,2)) / K(0,0);
    nbbox(2) = (bbox(2) - K(0,2)) / K(0,0);

    // for y 
    nbbox(1) = (bbox(1) - K(1,2)) / K(1,1);
    nbbox(3) = (bbox(3) - K(1,2)) / K(1,1);

    return nbbox;
}

// ref https://eigen.tuxfamily.org/bz/show_bug.cgi?id=448
template<typename Derived>
inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
{
	return ( (x - x).array() == (x - x).array()).all();
}

template<typename Derived>
inline bool check_nan(const Eigen::MatrixBase<Derived>& x)
{
  // returns false if there is nan 
  // since nan does not equal to itself 
	return ((x.array() == x.array())).all();
}

} // end namespace orcvio

#endif // MATH_UTILS_HPP
