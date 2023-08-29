// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNARITHM_HPP
#define OPENCV_CANNARITHM_HPP

#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{

/**
  @addtogroup cann
  @{
    @defgroup cannarithm Operations on Matrices
    @{
        @defgroup cannarithm_elem Per-element Operations
        @defgroup cannarithm_core Core Operations on Matrices
    @}
  @}
 */

//! @addtogroup cannarithm_elem
//! @{

/** @brief Computes a matrix-matrix or matrix-scalar sum.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::add cuda::add
 */
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask = noArray(), int dtype = -1,
                      AclStream& stream = AclStream::Null());
// This code should not be compiled nor analyzed by doxygen. This interface only for python binding
// code generation. add(InputArray, InputArray ...) can accept Scalar as its parametr.(Scalar -> Mat
// -> InputArray)
#ifdef NEVER_DEFINED
CV_EXPORTS_W void add(InputArray src1, Scalar src2, OutputArray dst, InputArray mask = noArray(),
                      int dtype = -1, AclStream& stream = AclStream::Null());
CV_EXPORTS_W void add(Scalar src1, InputArray src2, OutputArray dst, InputArray mask = noArray(),
                      int dtype = -1, AclStream& stream = AclStream::Null());
#endif

/** @brief Computes a matrix-matrix or matrix-scalar difference.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::subtract cuda::subtract
 */
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1,
                           AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void subtract(InputArray src1, Scalar src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1,
                           AclStream& stream = AclStream::Null());
CV_EXPORTS_W void subtract(Scalar src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1,
                           AclStream& stream = AclStream::Null());
#endif

/** @brief Computes a matrix-matrix or matrix-scalar per-element product.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param scale Optional scale factor.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::multiply cuda::multiply
 */
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2, OutputArray dst, float scale,
                           int dtype = -1, AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void multiply(InputArray src1, Scalar src2, OutputArray dst, float scale,
                           int dtype = -1, AclStream& stream = AclStream::Null());
CV_EXPORTS_W void multiply(Scalar src1, InputArray src2, OutputArray dst, float scale,
                           int dtype = -1, AclStream& stream = AclStream::Null());
#endif

/** @brief Computes a matrix-matrix or matrix-scalar division.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param scale Optional scale factor.
 * @param dtype Optional depth of the output array.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::divide cuda::divide
 */
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst, float scale,
                         int dtype = -1, AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void divide(InputArray src1, Scalar src2, OutputArray dst, float scale, int dtype = -1,
                         AclStream& stream = AclStream::Null());
CV_EXPORTS_W void divide(Scalar src1, InputArray src2, OutputArray dst, float scale, int dtype = -1,
                         AclStream& stream = AclStream::Null());
#endif

/** @brief Performs a per-element bitwise conjunction of two matrices (or of matrix and scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::bitwise_and cuda::bitwise_and
 */
CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_and(InputArray src1, Scalar src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
CV_EXPORTS_W void bitwise_and(Scalar src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#endif

/** @brief Performs a per-element bitwise disjunction of two matrices (or of matrix and scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::bitwise_or cuda::bitwise_or
 */
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2, OutputArray dst,
                             InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_or(InputArray src1, Scalar src2, OutputArray dst,
                             InputArray mask = noArray(), AclStream& stream = AclStream::Null());
CV_EXPORTS_W void bitwise_or(Scalar src1, InputArray src2, OutputArray dst,
                             InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#endif

/** @brief Performs a per-element bitwise exclusive or operation of two matrices (or of matrix and
 * scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::bitwise_xor cuda::bitwise_xor
 */
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_xor(InputArray src1, Scalar src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
CV_EXPORTS_W void bitwise_xor(Scalar src1, InputArray src2, OutputArray dst,
                              InputArray mask = noArray(), AclStream& stream = AclStream::Null());
#endif

/** @brief Performs a per-element bitwise inversion.
 * @param src1 First source matrix.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AclStream for the asynchronous version.
 * @sa cv::bitwise_not cuda::bitwise_not
 */
CV_EXPORTS_W void bitwise_not(InputArray src, OutputArray dst, InputArray mask = noArray(),
                              AclStream& stream = AclStream::Null());

/** @brief Computes the weighted sum of two arrays.

@param src1 First source array.
@param alpha Weight for the first array elements.
@param src2 Second source array of the same size and channel number as src1 .
@param beta Weight for the second array elements.
@param dst Destination array that has the same size and number of channels as the input arrays.
@param gamma Scalar added to each sum.
@param dtype Optional depth of the destination array. When both input arrays have the same depth,
dtype can be set to -1, which will be equivalent to src1.depth().
@param stream Stream for the asynchronous version.

The function addWeighted calculates the weighted sum of two arrays as follows:

\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)*
\texttt{beta} +  \texttt{gamma} )\f]

where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

@sa cv::addWeighted cv::cuda::addWeighted
 */
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2, double beta,
                              double gamma, OutputArray dst, int dtype = -1,
                              AclStream& stream = AclStream::Null());

//! @} cannarithm_elem

//! @addtogroup cannarithm_core
//! @{

/** @brief Makes a multi-channel matrix out of several single-channel matrices.

@param src Array/vector of source matrices.
@param n Number of source matrices.
@param dst Destination matrix.
@param stream AclStream for the asynchronous version.

@sa cv::merge cv::cuda::merge 
 */
CV_EXPORTS void merge(const AclMat* src, size_t n, OutputArray dst, AclStream& stream = AclStream::Null());
/** @overload */
CV_EXPORTS_W void merge(const std::vector<AclMat>& src, OutputArray dst, AclStream& stream = AclStream::Null());

/** @brief Copies each plane of a multi-channel matrix into an array.

@param src Source matrix.
@param dst Destination array/vector of single-channel matrices.
@param stream AclStream for the asynchronous version.

@sa cv::split cv::cuda::split
 */
CV_EXPORTS void split(InputArray src, AclMat* dst, AclStream& stream = AclStream::Null());
/** @overload */
CV_EXPORTS_W void split(InputArray src, CV_OUT std::vector<AclMat>& dst, AclStream& stream = AclStream::Null());

//! @} cannarithm_core

} // namespace cann
} // namespace cv

#endif /* OPENCV_CANNARITHM_HPP */
