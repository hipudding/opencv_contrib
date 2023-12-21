#ifndef TILING_KERNEL_H
#define TILING_KERNEL_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

inline __aicore__ int32_t Align32Ceil(int32_t n) { return ((n + 31) & ~31); }

inline __aicore__ int32_t Align32Floor(int32_t n) { return (n & ~31); }

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_BUF_LEN = 248 * 1024;

struct VectorTiling {
  __aicore__ inline void calculate(uint64_t totalLength, uint64_t blockNum,
                                   uint64_t blockIdx, uint32_t variableSize,
                                   uint32_t variableSetCount) {
    _totalLength = totalLength;
    _blockNum = blockNum;
    _blockIdx = blockIdx;
    _variableSize = variableSize;
    _variableSetCount = variableSetCount;
    _blockLength = 0;
    _blockOffset = 0;
    GetBlockLengthAndOffset();
    GetLoopLengthAndCount();
  }

  __aicore__ inline void GetBlockLengthAndOffset() {
    // Data should Align by 32B.
    uint32_t fullBlockLength = Align32Ceil(_totalLength / _blockNum);
    // Some core may get no data after Align32 Ceil.
    uint32_t fullBlockNum = _totalLength / fullBlockLength;
    uint32_t blockTailLength = _totalLength % fullBlockLength;

    if (_blockIdx < fullBlockNum) {
      _blockLength = fullBlockLength;
      _blockOffset = _blockIdx * _blockLength;
      // Last block must less than full block num.
    } else if (blockTailLength != 0 && _blockIdx == fullBlockNum) {
      _blockLength = blockTailLength;
      _blockOffset = _blockIdx * fullBlockLength;
    }
  }

  /**
   * @brief Get length for one loop and loop count.
   * Use as much UB buf as possible.
   */
  __aicore__ inline void GetLoopLengthAndCount() {
    _loopLength = Align32Floor(UB_BUF_LEN / _variableSize / _variableSetCount);
    _loopCount = _blockLength / _loopLength;
    _loopTailLength = _blockLength - (_loopLength * _loopCount);
  }

  uint64_t _totalLength;
  uint64_t _blockNum;
  uint64_t _blockIdx;
  uint32_t _variableSize;
  uint32_t _variableSetCount;
  uint32_t _blockLength;
  uint32_t _blockOffset;
  uint32_t _loopLength;
  uint32_t _loopCount;
  uint32_t _loopTailLength;
};

#endif  // TILING_KERNEL_H