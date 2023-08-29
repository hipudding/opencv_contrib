// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/cann_arithm.hpp"

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
    merge(channels, 3, m);

    cv::cann::setDevice(0);

    AclMat a1, a2, a3;
    a1.upload(m1);
    a2.upload(m2);
    a3.upload(m3);
    AclMat aclChannels[3] = {a1, a2, a3};

    AclMat ret;
    cv::cann::merge(aclChannels, 3, ret);

    Mat checker;
    ret.download(checker);

    EXPECT_MAT_NEAR(m, checker, 0.0);

    cv::cann::resetDevice();
}
} // namespace
} // namespace opencv_test