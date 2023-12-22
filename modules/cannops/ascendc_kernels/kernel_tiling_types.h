#ifndef KERNEL_TILING_H
#define KERNEL_TILING_H
/*
// ThresholdTypes is defined in opencv2/imgproc, This type is the only Symbol we need.
// Add imgproc to dependence is too heavy, use magic number instead.
enum ThresholdTypes {
    THRESH_BINARY     = 0,
    THRESH_BINARY_INV = 1,
    THRESH_TRUNC      = 2,
    THRESH_TOZERO     = 3,
    THRESH_TOZERO_INV = 4,
};
*/

#pragma pack(push, 8)
struct ThresholdOpencvTilingData
{
    int32_t maxVal;
    int32_t thresh;
    uint32_t totalLength;
    uint32_t threshType;
};
#pragma pack(pop)
#endif // KERNEL_TILING_H