#define _USE_MATH_DEFINES
#include <cmath>

#include <gtest/gtest.h>
#include <sophus/se2.hpp>

#include <orcvio/utils/EigenLevenbergMarquardt.h>

#include <orcvio/obj/ObjectFeatureInitializer.h>
#include <orcvio/obj/ObjectLM.h>

#ifdef HAVE_PYTHONLIBS
#include <orcvio/plot/matplotlibcpp.h>
constexpr bool DEBUG = false;
#else
constexpr bool DEBUG = false;
#endif


using namespace std;
using namespace Eigen;
using EigenLevenbergMarquardt::DenseFunctor;
namespace plt = matplotlibcpp;
using orcvio::ObjectLM;

// See https://github.com/madlib/eigen/blob/master/unsupported/test/levenberg_marquardt.cpp
struct lmder_functor : DenseFunctor<double>
{
    lmder_functor(void): DenseFunctor<double>(3,15) {}
    int operator()(const InputType &x, Eigen::Ref<ValueType> fvec) const
    {
        double tmp1, tmp2, tmp3;
        static const double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
        return 0;
    }

    int df(const InputType &x, Eigen::Ref<JacobianType> fjac) const
    {
        double tmp1, tmp2, tmp3, tmp4;
        for (int i = 0; i < values(); i++)
        {
            tmp1 = i+1;
            tmp2 = 16 - i - 1;
            tmp3 = (i>=8)? tmp2 : tmp1;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4 = tmp4*tmp4;
            fjac(i,0) = -1;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
        return 0;
    }
};


TEST(LevenbergMarquardt, test_levenberg_marquardt_der1) {
  int n=3, info;

  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  EigenLevenbergMarquardt::LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.lmder1(x);

  // check return value
  ASSERT_EQ(info, 1);
  ASSERT_EQ(lm.nfev(), 6);
  ASSERT_EQ(lm.njev(), 5);

  // check norm
  ASSERT_NEAR(lm.fvec().blueNorm(), 0.09063596, 1e-6);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  ASSERT_NEAR((x - x_ref).norm(), 0, 1e-6);

}

// See https://github.com/madlib/eigen/blob/master/unsupported/test/levenberg_marquardt.cpp
// Miniize (x - 10)^2
struct quadratic_func : DenseFunctor<double>
{
    quadratic_func(size_t input_size = 1,
                   size_t value_size = 1,
                   double minima = 10)
        : DenseFunctor<double>(input_size, value_size),
          minima_(minima)
    { }

    // fvec = (x - 10)
    int operator()(const InputType &x, Eigen::Ref<ValueType> fvec) const
    {
        fvec.block(0, 0, 1, 1) = (x.array() - minima_).matrix();
        return 0;
    }


    // fjac = \del (x-10) / \del x
    int df(const InputType &x, Eigen::Ref<JacobianType> fjac) const
    {

        fjac << 1;
        return 0;
    }

    double minima_;
};

TEST(LevenbergMarquardt, test_levenberg_marquardt_quadratic) {
  quadratic_func functor;
  int n=functor.m_inputs;

  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  EigenLevenbergMarquardt::LevenbergMarquardt<quadratic_func> lm(functor);
  auto info = lm.minimize(x);
  // check return value
  ASSERT_EQ(lm.info(), ComputationInfo::Success);
  ASSERT_EQ(info, EigenLevenbergMarquardt::LevenbergMarquardtSpace::CosinusTooSmall);
  ASSERT_EQ(lm.nfev(), 2);
  ASSERT_EQ(lm.njev(), 2);

  // check norm
  ASSERT_NEAR(lm.fvec().blueNorm(), 0, 1e-6);

  // check x
  VectorXd x_ref(n);
  x_ref << functor.minima_;
  ASSERT_NEAR((x - x_ref).norm(), 0, 1e-6);
}


template <typename Scalar, int NR=Dynamic, int NC=Dynamic>
struct EigenRange {
    const Eigen::Matrix<Scalar, NR, NC>& x_;
    const Eigen::Index i_;
    EigenRange(const Eigen::Matrix<Scalar, NR, NC>& x, const Eigen::Index i)
            : x_(x), i_(i)
    {}
    const Scalar* begin() const {
        return x_.col(i_).data();
    }
    const Scalar* end() const {
        return x_.col(i_).data() + x_.col(i_).size();
    }

    operator std::vector<Scalar> () const {
        std::vector<Scalar> vec(begin(), end());
        return vec;
    }
};

void
plot(const Eigen::MatrixX2d& x, const std::string& flags, const std::string& name = "") {
    EigenRange<double, Dynamic, 2> r0(x, 0);
    EigenRange<double, Dynamic, 2> r1(x, 1);
    if (name.length()) {
        plt::named_plot(name, static_cast<std::vector<double>>(r0),
                        static_cast<std::vector<double>>(r1),
                        flags);
    } else
        plt::plot(r0, r1, flags);
}


/**
 * @brief The fit_sin_curve struct
 *
 * Represents the loss function \sum_i |sin(\omega x_i) - y_i|
 */
struct fit_sin_curve : DenseFunctor<double>
{
    /**
     * @brief fit_sin_curve
     * @param points ((x_1, y_1), ..., (x_n, y_n))
     * @param omega
     */
    fit_sin_curve(const Eigen::MatrixX2d& points,
                  const double omega,
                  const double shift,
                  const double noise)
        : DenseFunctor<double>(2, points.rows()),
          params_(omega, shift),
          points_(points),
          noise_(noise)
    {}

    fit_sin_curve(const int npoints = 20, const double omega = 2.5, const double shift = 0.1, const double noise = 0.1)
        : DenseFunctor<double>(2, npoints),
          params_(omega, shift),
          points_(generatePoints(omega, shift, npoints, /*random=*/true, noise)),
          noise_(noise)
    { }

    /**
     * @brief operator ()
     * @return 0 on success
     * @param fvec: Output (sin(\omega x_i + shift) - y_i)
     */
    int operator ()(const InputType &params, Eigen::Ref<ValueType> fvec) const
    {
        assert (fvec.rows() == m_values);
        assert (params.rows() == m_inputs);
        assert (params.rows() == 2);
        auto x = points_.col(0).array();
        auto y = points_.col(1).array();
        Array<double, Dynamic, 1> angle = params(0,0) * x + params(1, 0);
        fvec.colwise() = (angle.sin() - y).matrix();
        return 0;
    }

    /**
     * @brief Compute the derivative of the non-linear function
     * @param omega
     * @return 0 on Success
     * @param fjac nvalues x ninputs
     */
    int df(const InputType &params, Eigen::Ref<JacobianType> fjac) const {
        assert (fjac.rows() == m_values);
        assert (fjac.cols() == m_inputs);
        assert (params.rows() == m_inputs);
        assert (params.rows() == 2);
        // Jac = x_i cos(\omega x_i)
        auto x = points_.col(0).array();
        fjac.col(0) = ((params(0,0) * x + params(1, 0)).cos() * x).matrix();
        fjac.col(1) = ((params(0,0) * x + params(1, 0)).cos()).matrix();
        return 0;
    }

    /**
     * @brief generatePoints
     * @param omega
     * @param n
     * @return
     */
    static Eigen::MatrixX2d generatePoints(double omega, double shift, int n, bool random = true, double noise = 0.1) {
        Eigen::MatrixX2d points = Eigen::MatrixX2d::Random(n, 2);
        if (!random)
            points.col(0).setLinSpaced(n, -1., 1.);
        points.col(1) = (omega * points.col(0).array() + shift).sin()
                + noise * Eigen::VectorXd::Random(n).array();
        return points;
    }

    const Vector2d params_;
    const Eigen::MatrixX2d points_;
    const double noise_;
};


TEST(LevenbergMarquardt, test_sin_fit) {
  fit_sin_curve functor;
  int n=functor.m_inputs;

  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setOnes(n, 1);

  // do the computation
  EigenLevenbergMarquardt::LevenbergMarquardt<fit_sin_curve> lm(functor);
  auto status = lm.minimize(x);
  if (DEBUG) {
      std::cout << "Exit with status: " << orcvio::LevenbergMarquardtStatusString(status) << "\n";
      plot(functor.points_, "r*", "input points");
      Eigen::MatrixX2d curve = fit_sin_curve::generatePoints(x(0,0), x(1,0), 10, /*random=*/false, /*noise=*/0);
      plot(curve, "g-", "fitted curve");
      plt::legend();
      plt::show();
  }
  // check return value
  ASSERT_EQ(lm.info(), ComputationInfo::Success);

//   EXPECT_LE(lm.nfev(), 7);
//   EXPECT_LE(lm.njev(), 7);
  EXPECT_LE(lm.nfev(), 9);
  EXPECT_LE(lm.njev(), 8);

  double margin = sqrt(functor.noise_);
  // check norm
  ASSERT_NEAR(lm.fvec().blueNorm(), margin, margin);

  // check x
  ASSERT_NEAR(x(0,0), functor.params_(0), sqrt(functor.noise_));
  ASSERT_NEAR(x(1,0), functor.params_(1), sqrt(functor.noise_));
}

Sophus::SE2d operator+(const Sophus::SE2d& x,const Eigen::Vector3d& dx) {
    return Sophus::SE2d::exp(dx) * x;
}

std::ostream& operator<<(std::ostream& o, const Sophus::SE2d& x) {
    o << "SE2d(" << x.log() << ")";
    return o;
}

struct SE2 : Sophus::SE2d {
  template <typename ... Args>
  SE2(Args ... args) : Sophus::SE2d(args ...)
  {}
  int size() {
    return Sophus::SE2d::DoF;
  }
};


/**
 * @brief The find_se2_transform
 *
 * Find the world to object transformation, given 2D points in two world cordintates
 *
 * Represents the loss function \f$ \sum_i |wTo x_o - x_w| \f$
 */
struct find_se2_transform : DenseFunctor<double, Eigen::Dynamic, Eigen::Dynamic, SE2>
{
    typedef DenseFunctor<double, Eigen::Dynamic, Eigen::Dynamic, SE2> BaseDenseFunctor;
    /**
     * @brief find_se2_transform
     * @param x_o ((x_1, y_1), ..., (x_n, y_n)) points in object frame
     * @param x_w ((x_1, y_1, ..., (x_n, y_n)) points in world frame
     * @param theta
     * @param shift
     */
    find_se2_transform(const Eigen::MatrixX2d& x_o,
                  const Eigen::MatrixX2d& x_w,
                  const double theta,
                  const Eigen::Vector2d& shift)
        : BaseDenseFunctor(3, 2*x_o.rows()),
          theta_(theta),
          shift_(shift),
          x_o_(x_o),
          x_w_(x_w)
    {}

    find_se2_transform(const int npoints,
                       const Eigen::Vector2d& shift,
                       const double theta)
        : BaseDenseFunctor(3, 2*npoints),
          theta_(theta),
          shift_(shift),
          x_o_(Eigen::MatrixX2d::Random(npoints, 2)),
          x_w_(transformPoints(theta, shift, x_o_))
    {
    }

    /**
     * @brief operator ()
     * @param[in] wTo
     * @param[out] fvec: Output
     *
     * @return 0 on success
     */
    int operator ()(const InputType &wTo, Eigen::Ref<ValueType> fvec) const
    {
        assert (fvec.rows() == m_values);
        Eigen::MatrixX2d fvec_mat =  (
                            ((wTo.rotationMatrix() * x_o_.transpose()).colwise() + wTo.translation())
                            - x_w_.transpose()
                            ).transpose();
        fvec.noalias() = Map<ValueType>(fvec_mat.data(), fvec.rows(), 1);
        return 0;
    }

    /**
     * @brief Compute the derivative of the non-linear function
     * @param omega
     * @return 0 on Success
     * @param fjac nvalues x ninputs
     */
    int df(const InputType& wTo, Eigen::Ref<JacobianType> fjac) const {
        assert (fjac.rows() == m_values);
        assert (fjac.cols() == m_inputs);
        double theta = wTo.log()(2);
        fjac.col(0).head(x_o_.rows()).setConstant(1);
        fjac.col(0).tail(x_o_.rows()).setConstant(0);
        fjac.col(1).head(x_o_.rows()).setConstant(0);
        fjac.col(1).tail(x_o_.rows()).setConstant(1);
        fjac.col(2).head(x_o_.rows()) =
            - x_o_.col(0).array() * cos(theta) - x_o_.col(1).array() * sin(theta);
        fjac.col(2).tail(x_o_.rows()) =
            x_o_.col(0).array() * sin(theta) - x_o_.col(1).array() * cos(theta);
        return 0;
    }

    /**
     * @brief scaled_norm
     *
     * LM solves the following problem iteratively
     *
     * min || f + Jp || s.t. ||Dp|| < delta < XTOL ||Dx||
     *
     * where delta < XTOL || Dx || is used as a relative error chceck.
     *
     * This function computes || Dx ||
     *
     * @param D
     * @param wTo
     * @return || D log(wTo) ||
     */
    Scalar scaled_norm(const DiffType& D, const InputType& wTo) const {
        DiffType x = wTo.log();
        return D.cwiseProduct(x).stableNorm();
    }

    /**
     * @brief generatePoints
     * @param omega
     * @param n
     * @return
     */
    static Eigen::MatrixX2d transformPoints(
            double theta,  const Eigen::Vector2d& shift, const Eigen::MatrixX2d&  x_o, double noise = 0.05)
    {
        Sophus::SE2d wTo(theta, shift);
        Eigen::MatrixX2d x_w = ((wTo.rotationMatrix() * x_o.transpose()).colwise()
                + wTo.translation()).transpose()
                + noise * Eigen::MatrixX2d::Random(x_o.rows(), 2);
        return x_w;
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    const double theta_;
    const Eigen::Vector2d& shift_;
    const Eigen::MatrixX2d x_o_;
    const Eigen::MatrixX2d x_w_;
};


TEST(LevenbergMarquardt, test_se2_transform) {
    double theta = M_PI / 3;
    Eigen::Vector2d shift;
    shift << 2.1, -1.0;
    int npoints = 20;
    find_se2_transform functor(npoints, shift, theta);

    Eigen::Vector2d shift_0{0.1,  0.1};
    double theta_0 = 0.1;
    SE2 x(theta_0, shift_0);

    // do the computation
    EigenLevenbergMarquardt::LevenbergMarquardt<find_se2_transform> lm(functor);
    lm.setFactor(10);
    auto status = lm.minimize(x);
    // check return value
    ASSERT_EQ(lm.info(), ComputationInfo::Success);
    if (DEBUG) {
        std::cout << "Exit with status: " << orcvio::LevenbergMarquardtStatusString(status) << "\n";

        plot(functor.x_o_, "b*", "original points");
        double theta_est = x.log()(2);
        auto  shift_est = x.translation();
        auto x_w = find_se2_transform::transformPoints(theta_est, shift_est, functor.x_o_);
        plot(x_w, "r+", "transform with estimate");
        plot(functor.x_w_, "go", "ground truth");
        plt::legend();
        plt::show();
    }

    double noise = npoints * 2 * 0.05;
    // check norm
    ASSERT_NEAR(lm.fvec().blueNorm(), noise, noise);

    // check x
    Eigen::Vector3d xvec = x.log();
    ASSERT_NEAR(xvec(2), functor.theta_, noise);
    ASSERT_NEAR(xvec(0), functor.shift_(0), noise);
    ASSERT_NEAR(xvec(1), functor.shift_(1), noise);
}


