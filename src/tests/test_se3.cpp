#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

#include <orcvio/utils/EigenLevenbergMarquardt.h>
#include <orcvio/utils/EigenNumericalDiff.h>
#include <orcvio/utils/se3_ops.hpp>

using namespace std;
using namespace orcvio;

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::MatrixX4d;
using Eigen::Matrix4d;
using Eigen::Map;
using Eigen::RowMajor;
using Eigen::Dynamic;
using EigenNumericalDiff::NumericalDiff;
using EigenLevenbergMarquardt::DenseFunctor;

TEST(SE3UtilsTest, test_skew_batch) 
{

    vector<Eigen::Matrix<double, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 1> > > a;
    Eigen::Vector3d w(1.0, 2.0, 3.0);
    a.push_back(w);
    a.push_back(w);

    vector<Eigen::Matrix<double, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 3> > > w_hat = skew(a);
    Eigen::Vector3d zero_vector1 = w_hat[0] * w;
    Eigen::Vector3d zero_vector2 = w_hat[1] * w;

    EXPECT_DOUBLE_EQ(zero_vector1.norm(), 0.0);
    EXPECT_DOUBLE_EQ(zero_vector2.norm(), 0.0);

    return;

}

TEST(SE3UtilsTest, test_inversePose_batch) 
{

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd rotation_vector(3.14 / 4, Eigen::Vector3d(0, 0, 1));
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1, 3, 4));

    vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4> > > T_batch;
    T_batch.push_back(T.matrix());
    T_batch.push_back(T.matrix());
    
    vector<Eigen::Matrix<double, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 4, 4> > > iT = inversePose(T_batch);
    ASSERT_TRUE(iT[0] == T.inverse().matrix());
    ASSERT_TRUE(iT[1] == T.inverse().matrix());

}

TEST(SE3UtilsTest, test_project_object_points)
{
    double f = 0.25, x = 2,  y = 2, X = 4.00, Y = 4.00, Z = 0.5;
    Eigen::Matrix<double, 3, 4> P;
    P <<
      f, 0, x, 0,
      0, f, y, 0,
      0, 0, 1, 0;
    Eigen::Matrix4d wTo = Eigen::Matrix4d::Identity();
    Eigen::MatrixX4d points_w(1,4);
    points_w << X, Y, Z, 1;
    Eigen::MatrixX2d uvs = orcvio::project_object_points(P, wTo, points_w);
    Eigen::MatrixX2d uvs_expected(1, 2); uvs_expected << f * X / Z  + x, f * Y  / Z + y;
    ASSERT_EQ(uvs, uvs_expected);
}

TEST(SE3UtilsTest, test_project_object_points_data) {

  auto name2mat = cv::hdf::open("src/tests/data/test_error_feature_quadric.h5");
  auto const zs_e = dsread(name2mat, "zs");
  auto const S_e = dsread(name2mat, "S");
  auto const T_e = dsread(name2mat, "T");
  auto const M_e = dsread(name2mat, "M");
  auto const errors_e = dsread(name2mat, "error");
  auto P = S_e.topRows<3>();
  Eigen::MatrixX2d uvs = orcvio::project_object_points(P, T_e, M_e);
  auto uvs_true = Map<const Matrix<double, Dynamic, 2, RowMajor>>(errors_e.data(), zs_e.rows(), zs_e.cols()) + zs_e;
  ASSERT_NEAR((uvs - uvs_true).norm(), 0, 1e-6)
          << "uvs : \n" << uvs << "\n"
          << "uvs_true : \n" << uvs_true << "\n";
}



struct SE3d : Sophus::SE3d {
  template <typename A1>
  SE3d(A1& a) : Sophus::SE3d(a) {}
  typedef Eigen::Index Index;
  Index size() const {
    return Sophus::SE3d::DoF;
  }
  Scalar operator[](const Index& i) const {
    return Sophus::SE3d::log()[i];
  }
};

Sophus::SE3d operator+(const Sophus::SE3d& x, const Eigen::Matrix<double, 6, 1>& dx) {
  Sophus::SE3d ret(Sophus::SE3d::exp(dx).matrix() * x.matrix());
  return ret;
}

Sophus::SE3d operator-(const Sophus::SE3d& x, const Eigen::Matrix<double, 6, 1>& dx) {
  Sophus::SE3d ret(Sophus::SE3d::exp(-dx).matrix() * x.matrix());
  return ret;
}

Sophus::SE3d operator-=(Sophus::SE3d& x, const Eigen::Matrix<double, 6, 1>& dx) {
  // x = x - dx;
  x = Sophus::SE3d::exp(-dx) * x;
  return x;
}

Sophus::SE3d operator+=(Sophus::SE3d& x, const Eigen::Matrix<double, 6, 1>& dx) {
  // x = x + dx;
  x = Sophus::SE3d::exp(dx) * x;
  return x;
}


struct test_project_object_points {
  constexpr static int InputsAtCompileTime = 6;
  constexpr static int ValuesAtCompileTime = 2;
  typedef SE3d InputType;
  typedef typename Eigen::Matrix<double, InputsAtCompileTime, 1> DiffType;
  typedef double Scalar;
  typedef typename Eigen::Vector2d ValueType;
  typedef typename Eigen::Matrix<double, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  test_project_object_points(const Matrix<double, 3, 4>& P,
                             const MatrixX4d& points_w)
    : P_(P),
      points_w_(points_w),
      m_inputs(InputsAtCompileTime),
      m_values(ValuesAtCompileTime)
  {};

  int operator() (const InputType& x, ValueType& fvec) const {
    const Matrix4d wTo = x.matrix();
    Eigen::MatrixX2d uvs = orcvio::project_object_points(P_, wTo, points_w_);
    assert (fvec.size() == uvs.size());
    // Reminder: Eigen is column major
    fvec.noalias() = Map<ValueType>(uvs.data(), fvec.rows(), fvec.cols());
    return 0;
  }

  Eigen::Index values() const {
    return m_values;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  const Matrix<double, 3, 4>& P_;
  const MatrixX4d& points_w_;
  Eigen::Index m_inputs, m_values;
};


TEST(SE3UtilsTest, test_project_object_points_df_object)
{
    Eigen::Matrix<double, 3, 4> P;
    P.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d wTo = Eigen::Matrix4d::Identity();
    Eigen::MatrixX4d points_w(1,4);
    double X = 4.00, Y = 4.00, Z = 4.00;
    points_w << X, Y, Z, 1;
    bool use_left_perturbation_flag = true;
    Eigen::Matrix<double, Eigen::Dynamic, 6>  jacobians =
      project_object_points_df_object(P, wTo, points_w,
                                      use_left_perturbation_flag);

    test_project_object_points proj_object_pts(P, points_w);
    NumericalDiff<test_project_object_points> num_proj_object_pts(proj_object_pts);
    Eigen::Matrix<double, 2, 6> num_jac;
    SE3d wTo_SE3(wTo);
    num_proj_object_pts.df(wTo_SE3, num_jac);
    auto first_jac = jacobians.block<2,6>(0,0);
    double atol = 1e-4;
    double rtol = 1e-3;
    ASSERT_TRUE(((first_jac - num_jac).array().abs()
                 <= (atol + rtol * num_jac.array().abs())).all())
        << "Diff: \n" << (first_jac - num_jac).array().abs() << "\n"
        << "Tol: \n" << (atol + rtol * num_jac.array().abs()) << "\n"
        << "evaluated : \n" << first_jac << "\n"
        << "numerical : \n" << num_jac << "\n";
}


