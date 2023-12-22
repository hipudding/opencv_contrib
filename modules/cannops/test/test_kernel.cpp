#include "test_precomp.hpp"
#include "acl/acl.h"

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

    uint8_t* tilingDevice;
    aclrtMalloc((void**)&tilingDevice, sizeof(ThresholdOpencvTilingData), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(tilingDevice, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
    
    uint32_t ret = ACLRT_LAUNCH_KERNEL(threshold_opencv)(8, nullptr, npuImg32F.data.get(), npuChecker.data.get(), tilingDevice);
    aclError err = aclrtSynchronizeStream(nullptr);

    npuChecker.download(npuRet);
    EXPECT_MAT_NEAR(cpuRet, npuRet, 10.0f);
    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
