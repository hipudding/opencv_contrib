// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/cann_arithm.hpp"
#include <vector>

namespace opencv_test
{
namespace
{
TEST(CORE, MERGE)
{
    Mat m1 = (Mat_<uchar>(2, 2) << 1, 4, 7, 10);
    Mat m2 = (Mat_<uchar>(2, 2) << 2, 5, 8, 11);
    Mat m3 = (Mat_<uchar>(2, 2) << 3, 6, 9, 12);
    Mat channels[3] = {m1, m2, m3};
    Mat m;
    cv::merge(channels, 3, m);

    cv::cann::setDevice(0);

    AclMat a1, a2, a3;
    a1.upload(m1);
    a2.upload(m2);
    a3.upload(m3);
    AclMat aclChannels[3] = {a1, a2, a3};
    std::vector<AclMat> aclChannelsVector;
    aclChannelsVector.push_back(a1);
    aclChannelsVector.push_back(a2);
    aclChannelsVector.push_back(a3);

    AclMat ret1, ret2;
    cv::cann::merge(aclChannels, 3, ret1);
    cv::cann::merge(aclChannelsVector, ret2);

    Mat checker1, checker2;
    ret1.download(checker1);
    ret2.download(checker2);

    EXPECT_MAT_NEAR(m, checker1, 0.0);
    EXPECT_MAT_NEAR(m, checker2, 0.0);

    cv::cann::resetDevice();
}

TEST(CORE, SPLIT)
{
    char d[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Mat m(2, 2, CV_8UC3, d);
    Mat channels[3];
    cv::split(m, channels);

    cv::cann::setDevice(0);

    AclMat aclMat;
    aclMat.upload(m);

    AclMat aclChannels[3];
    std::vector<AclMat> aclChannelsVector;

    cv::cann::split(aclMat, aclChannels);
    cv::cann::split(aclMat, aclChannelsVector);

    Mat checker1[3], checker2[3];
    aclChannels[0].download(checker1[0]);
    aclChannels[1].download(checker1[1]);
    aclChannels[2].download(checker1[2]);

    aclChannelsVector[0].download(checker2[0]);
    aclChannelsVector[1].download(checker2[1]);
    aclChannelsVector[2].download(checker2[2]);

    EXPECT_MAT_NEAR(channels[0], checker1[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker1[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker1[2], 0.0);

    EXPECT_MAT_NEAR(channels[0], checker2[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker2[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker2[2], 0.0);

    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
