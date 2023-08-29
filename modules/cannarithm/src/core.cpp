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

    // CANN ops can't deal with gaps between rows.
    CV_Assert(src->isContinuous());

    // all matrix must have same size and type
    for (size_t i = 1; i < n; i++)
    {
        CV_Assert(src[i].depth() == depth && src[i].channels() == 1);
        CV_Assert(src[i].rows == rows && src[i].cols == cols);
        CV_Assert(src[i].isContinuous());
    }

    AclMat dst = getOutputMat(_dst, rows, cols, CV_MAKE_TYPE(depth, n));

    // considering no gaps between rows, the memory size between NCHW and NHWC is the same.
    // TODO: AclMat's format is NHWC, fix it in the furture.
    AclMat NCHWMat(rows, cols, CV_MAKE_TYPE(depth, n));
    CV_Assert(NCHWMat.isContinuous() && dst.isContinuous());

    uchar* ptr = NCHWMat.data;
    size_t remainSize = rows * NCHWMat.step;
    aclrtStream rawStream = AclStreamAccessor::getStream(stream);

    for (size_t i = 0; i < n; i++)
    {
        size_t copySize = rows * src[i].step;
        if (rawStream == nullptr)
            CV_ACL_SAFE_CALL(
                aclrtMemcpy(ptr, remainSize, src[i].data, copySize, ACL_MEMCPY_DEVICE_TO_DEVICE));
        else
            CV_ACL_SAFE_CALL(aclrtMemcpyAsync(ptr, remainSize, src[i].data, copySize,
                                              ACL_MEMCPY_DEVICE_TO_DEVICE, rawStream));
        ptr += copySize;
        remainSize -= copySize;
    }

    transData(NCHWMat, dst, "NCHW", "NHWC", stream);
}

void merge(const std::vector<AclMat>& src, OutputArray dst, AclStream& stream)
{
    merge(&src[0], src.size(), dst, stream);
}

void split(InputArray _src, AclMat* dst, AclStream& stream)
{
    // TODO implemnt InputArray for AclMat.
    if (/*src.empty() ||*/ dst == nullptr)
    {
        return;
    }

    AclMat src = getInputMat(_src, stream);
    AclMat NCHWMat(src.size(), src.type());
    CV_Assert(src.isContinuous() && NCHWMat.isContinuous());

    transData(src, NCHWMat, "NHWC", "NCHW", stream);

    uchar* ptr = NCHWMat.data;
    aclrtStream rawStream = AclStreamAccessor::getStream(stream);

    for (int i = 0; i < NCHWMat.channels(); i++)
    {
        dst[i].create(NCHWMat.rows, NCHWMat.cols, CV_MAKE_TYPE(NCHWMat.depth(), 1));
        CV_Assert(dst->isContinuous());
        size_t copySize = dst[i].rows * dst[i].step;

        if (rawStream == nullptr)
            CV_ACL_SAFE_CALL(
                aclrtMemcpy(dst[i].data, copySize, ptr, copySize, ACL_MEMCPY_DEVICE_TO_DEVICE));
        else
            CV_ACL_SAFE_CALL(aclrtMemcpyAsync(dst[i].data, copySize, ptr, copySize,
                                              ACL_MEMCPY_DEVICE_TO_DEVICE, rawStream));
        ptr += copySize;
    }
}

void split(InputArray _src, std::vector<AclMat>& dst, AclStream& stream)
{
    // TODO implement channels in InputArray.
    AclMat src = getInputMat(_src);
    dst.resize(src.channels());
    split(_src, &dst[0], stream);
}

} // namespace cann
} // namespace cv
