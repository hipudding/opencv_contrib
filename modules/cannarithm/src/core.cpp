// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <iostream>

namespace cv
{
namespace cann
{
void merge(const AclMat* src, size_t n, OutputArray _dst, AclStream& stream)
{
    if (src == nullptr || n == 0)
        return;

    int depth = src->depth();
    int rows = src->rows;
    int cols = src->cols;
    bool isContinuous = src->isContinuous();

    // all matrix must have same size and type
    for (size_t i = 1; i < n; i++)
    {
        CV_Assert(src[i].depth() == depth && src[i].channels() == 1);
        CV_Assert(src[i].rows == rows && src[i].cols == cols);
        isContinuous = isContinuous && src[i].isContinuous();
    }

    // TODO use origin ptr instead of AclMat, in case of gaps between rows.
    AclMat dst = getOutputMat(_dst, rows, cols, CV_MAKE_TYPE(depth, n));
    AclMat NCHWMat(rows, cols, CV_MAKE_TYPE(depth, n));
    isContinuous = isContinuous && NCHWMat.isContinuous();

    uchar* ptr = NCHWMat.data;
    size_t remainSize = rows * NCHWMat.step;
    // no gap between rows, copy channel by channel;
    if (isContinuous)
    {
        for (size_t i = 0; i < n; i++)
        {
            size_t copySize = rows * src[i].step;
            CV_ACL_SAFE_CALL(
                aclrtMemcpy(ptr, remainSize, src[i].data, copySize, ACL_MEMCPY_DEVICE_TO_DEVICE));
            ptr += copySize;
            remainSize -= copySize;
        }
    }
    // step is not same, should copy line by line.
    // TODO: AscenCL op can't process with gap between lines!!!
    else
    {
        for (size_t i = 0; i < n; i++)
        {
            uchar* srcPtr = src[i].data;
            for (int j = 0; j < src[i].rows; j++)
            {
                CV_ACL_SAFE_CALL(aclrtMemcpy(ptr, remainSize, srcPtr, src[i].elemSize() * cols,
                                             ACL_MEMCPY_DEVICE_TO_DEVICE));
                ptr += NCHWMat.step;
                remainSize -= NCHWMat.step;
                srcPtr += src[1].step;
            }
        }
    }

    transData(NCHWMat, dst, "NCHW", "NHWC", stream);
}
} // namespace cann
} // namespace cv