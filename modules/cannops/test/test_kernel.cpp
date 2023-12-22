#include "test_precomp.hpp"
#include "opencv2/cann_call.hpp"

namespace opencv_test
{
namespace{

TEST(ASCENDC_KERNEL, THRESHOLD)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat cpuRet, npuRet;

    Mat img32F = randomMat(512, 512, CV_32FC3, 0.0f, 255.0f);
    cv::threshold(img32F, cpuRet, 200, 255, 2);

    AscendMat npuImg32F, npuChecker;
    npuImg32F.upload(img32F);

    npuChecker.create(npuImg32F.rows, npuImg32F.cols, npuImg32F.type());

    ThresholdOpencvTilingData tiling;
    tiling.maxVal = 255;
    tiling.thresh = 200;
    tiling.totalLength = img32F.rows * img32F.cols * img32F.channels();
    tiling.threshType = 2;

    kernel_launch(aclrtlaunch_threshold_opencv, AscendStream::Null(), tiling, npuImg32F.data.get(), npuChecker.data.get());

    npuChecker.download(npuRet);
    EXPECT_MAT_NEAR(cpuRet, npuRet, 10.0f);
    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
