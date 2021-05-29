#include <gtest/gtest.h>

#include <orcvio/obj/ObjectFeatureInitializer.h>

using namespace std;
using namespace orcvio;
using namespace Eigen;

// TEST(TestSuite, testCase1)
TEST(KabschTest, test_random) 
{
    // Create datasets with known transform
    Eigen::Matrix3Xd in(3, 100), out(3, 100);
    Eigen::Quaternion<double> Q(1, 3, 5, 2);
    Q.normalize();

    Eigen::Matrix3d R = Q.toRotationMatrix();
    double scale = 2.0;

    for (int row = 0; row < in.rows(); row++) 
    {
        for (int col = 0; col < in.cols(); col++) 
        {
            in(row, col) = log(2*row + 10.0)/sqrt(1.0*col + 4.0) + sqrt(col*1.0)/(row + 1.0);
        }
    }

    Eigen::Vector3d S;
    S << -5, 6, -27;

    for (int col = 0; col < in.cols(); col++)
        out.col(col) = scale * R * in.col(col) + S;

    Eigen::Matrix4d Trans = findTransform(in, out);

    Eigen::Matrix3d Trans_R = Trans.block<3,3>(0,0);
    Eigen::Vector3d Trans_t = Trans.block<3,1>(0,3);

    // See if we got the transform we expected
    if ( (scale * R - Trans_R).cwiseAbs().maxCoeff() > 1e-13 || (S - Trans_t).cwiseAbs().maxCoeff() > 1e-13)
        throw "Could not determine the affine transform accurately enough";

}

// TEST(TestSuite, testCase1)
TEST(KabschTest, test_planar) 
{
    // Create datasets with known transform
    Eigen::Matrix3Xd in(3, 4), out(3, 4);
    Eigen::Quaternion<double> Q(1, 3, 5, 2);
    Q.normalize();

    Eigen::Matrix3d R = Q.toRotationMatrix();
    double scale = 2.0;

    in(0, 0) = -1.25;
    in(1, 0) = 0;
    in(2, 0) = -1.25;

    in(0, 1) = 1.25;
    in(1, 1) = 0;
    in(2, 1) = -1.25;

    in(0, 2) = 1.25;
    in(1, 2) = 0;
    in(2, 2) = 1.25;

    in(0, 3) = -1.25;
    in(1, 3) = 0;
    in(2, 3) = 1.25;

    Eigen::Vector3d S;
    S << -5, 6, -27;

    for (int col = 0; col < in.cols(); col++)
        out.col(col) = scale * R * in.col(col) + S;

    Eigen::Matrix4d Trans = findTransform(in, out);

    Eigen::Matrix3d Trans_R = Trans.block<3,3>(0,0);
    Eigen::Vector3d Trans_t = Trans.block<3,1>(0,3);

    // See if we got the transform we expected
    if ( (scale * R - Trans_R).cwiseAbs().maxCoeff() > 1e-13 || (S - Trans_t).cwiseAbs().maxCoeff() > 1e-13)
        throw "Could not determine the affine transform accurately enough";

}
