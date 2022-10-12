#include <cuda/cudaTypeDef.cuh>
#include <cuda_runtime_api.h>

namespace sl {
    namespace phaseSolver {
        namespace cudaFunc {
            __global__ void atan2M_FourStepSixGray(
                    const cv::cuda::PtrStep<uchar> shift_0_,
                    const cv::cuda::PtrStep<uchar> shift_1_,
                    const cv::cuda::PtrStep<uchar> shift_2_,
                    const cv::cuda::PtrStep<uchar> shift_3_,
                    const int rows, const int cols,
                    cv::cuda::PtrStep<float> wrapImg, cv::cuda::PtrStep<float> averageImg,
                    cv::cuda::PtrStep<float> conditionImg) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    wrapImg.ptr(y)[x] = cuda::std::atan2f(shift_3_.ptr(y)[x] - shift_1_.ptr(y)[x], shift_0_.ptr(y)[x] - shift_2_.ptr(y)[x]);
                    averageImg.ptr(y)[x] = (shift_0_.ptr(y)[x] + shift_1_.ptr(y)[x] + shift_2_.ptr(y)[x] + shift_3_.ptr(y)[x]) / 4.0f;
                    conditionImg.ptr(y)[x] = cuda::std::sqrt((cuda::std::pow((shift_3_.ptr(y)[x] - shift_1_.ptr(y)[x]), 2) + cuda::std::pow((shift_0_.ptr(y)[x] - shift_2_.ptr(y)[x]), 2))) / 2.0f;
                }
            }

            __global__ void getUnwrapImg_FourStepSixGray(
                    const cv::cuda::PtrStep<uchar> Gray_0_,
                    const cv::cuda::PtrStep<uchar> Gray_1_,
                    const cv::cuda::PtrStep<uchar> Gray_2_,
                    const cv::cuda::PtrStep<uchar> Gray_3_,
                    const cv::cuda::PtrStep<uchar> Gray_4_,
                    const cv::cuda::PtrStep<uchar> Gray_5_,
                    const int rows, const int cols,
                    const cv::cuda::PtrStep<float> averageImg,
                    const cv::cuda::PtrStep<float> conditionImg,
                    const cv::cuda::PtrStep<float> wrapImg,
                    cv::cuda::PtrStep<float> unwrapImg) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    if (conditionImg.ptr(y)[x] < 20.0) {
                        unwrapImg.ptr(y)[x] = -5.f;
                    } else {
                        const uchar bool_0 = Gray_0_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                        const uchar bool_1 = Gray_1_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                        const uchar bool_2 = Gray_2_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                        const uchar bool_3 = Gray_3_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                        const uchar bool_4 = Gray_4_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                        const uchar bool_5 = Gray_5_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                        uchar bit_5 = bool_0 ^ 0;
                        uchar bit_4 = bool_1 ^ bit_5;
                        uchar bit_3 = bool_2 ^ bit_4;
                        uchar bit_2 = bool_3 ^ bit_3;
                        uchar bit_1 = bool_4 ^ bit_2;
                        uchar bit_0 = bool_5 ^ bit_1;
                        const int K2 = cuda::std::floorf(((bit_5 * 32) + (bit_4 * 16) + (bit_3 * 8) + (bit_2 * 4) + (bit_1 * 2) + bit_0 + 1) / 2);
                        bit_4 = bool_0 ^ 0;
                        bit_3 = bool_1 ^ bit_4;
                        bit_2 = bool_2 ^ bit_3;
                        bit_1 = bool_3 ^ bit_2;
                        bit_0 = bool_4 ^ bit_1;
                        const int K1 = (bit_4 * 16) + (bit_3 * 8) + (bit_2 * 4) + (bit_1 * 2) + bit_0;
                        if (wrapImg.ptr(y)[x] <= (-CV_PI / 2))
                            unwrapImg.ptr(y)[x] = wrapImg.ptr(y)[x] + CV_PI * 2 * K2;
                        if (wrapImg.ptr(y)[x] > (-CV_PI / 2) && wrapImg.ptr(y)[x] < (CV_PI / 2))
                            unwrapImg.ptr(y)[x] = wrapImg.ptr(y)[x] + CV_PI * 2 * K1;
                        if (wrapImg.ptr(y)[x] >= (CV_PI / 2))
                            unwrapImg.ptr(y)[x] = wrapImg.ptr(y)[x] + CV_PI * 2 * (K2 - 1);
                    }
                }
            }

            __device__ void sortData(float *data, const int size) {
                bool flag = true;
                for (int i = 0; i < size - 1; i++) {
                    for (int j = 0; j < size - i - 1; j++) {
                        if (data[j] > data[j + 1]) {
                            float temp = data[j];
                            data[j] = data[j + 1];
                            data[j + 1] = temp;
                        }
                    }
                }
            }

            __global__ void kMeansCluster(
                    const cv::cuda::PtrStep<float> src,
                    const cv::cuda::PtrStep<float> conditionImg,
                    const int rows, const int cols,
                    cv::cuda::PtrStep<float> threshodVal,
                    cv::cuda::PtrStep<float> threshodAdd,
                    cv::cuda::PtrStep<float> count) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    if (conditionImg.ptr(y)[x] > 10.f) {
                        float minDistance = 5.f;
                        int family = 0;
                        for (int i = 0; i < 4; i++) {
                            float distance = cuda::std::abs(src.ptr(y)[x] - threshodVal.ptr(i)[0]);
                            if (distance < minDistance) {
                                minDistance = distance;
                                family = i;
                            }
                        }
                        atomicAdd(&threshodAdd.ptr(family)[0], src.ptr(y)[x]);
                        atomicAdd(&count.ptr(family)[0], 1.f);
                    }
                }

                __syncthreads();

                if (x == 1 && y == 1) {
                    if (count.ptr(0)[0] < 100)
                        return;
                    if (count.ptr(0)[0] != 0) {
                        float val = threshodAdd.ptr(0)[0] / count.ptr(0)[0];
                        if (std::abs(val + 1.f) < 0.3)
                            threshodVal.ptr(0)[0] = threshodAdd.ptr(0)[0] / count.ptr(0)[0];
                    }
                    threshodAdd.ptr(0)[0] = 0;
                    count.ptr(0)[0] = 0;
                } else if (x == 1 && y == 2) {
                    if (count.ptr(1)[0] < 100)
                        return;
                    if (count.ptr(1)[0] != 0) {
                        float val = threshodAdd.ptr(1)[0] / count.ptr(1)[0];
                        if (std::abs(val + 0.33f) < 0.3)
                            threshodVal.ptr(1)[0] = threshodAdd.ptr(1)[0] / count.ptr(1)[0];
                    }
                    threshodAdd.ptr(1)[0] = 0;
                    count.ptr(1)[0] = 0;
                } else if (x == 1 && y == 3) {
                    if (count.ptr(2)[0] < 100)
                        return;
                    if (count.ptr(2)[0] != 0) {
                        float val = threshodAdd.ptr(2)[0] / count.ptr(2)[0];
                        if (std::abs(val - 0.33f) < 0.3)
                            threshodVal.ptr(2)[0] = threshodAdd.ptr(2)[0] / count.ptr(2)[0];
                    }
                    threshodAdd.ptr(2)[0] = 0;
                    count.ptr(2)[0] = 0;

                } else if (x == 1 && y == 4) {
                    if (count.ptr(3)[0] < 100)
                        return;
                    if (count.ptr(3)[0] != 0) {
                        float val = threshodAdd.ptr(3)[0] / count.ptr(3)[0];
                        if (std::abs(val - 1.f) < 0.3)
                            threshodVal.ptr(3)[0] = threshodAdd.ptr(3)[0] / count.ptr(3)[0];
                    }
                    threshodAdd.ptr(3)[0] = 0;
                    count.ptr(3)[0] = 0;
                }
            }

            __global__ void medianFilter(
                    const cv::cuda::PtrStep<float> src,
                    const cv::cuda::PtrStep<float> conditionImgCopy,
                    const int kSize, const int rows, const int cols,
                    cv::cuda::PtrStep<float> dst, cv::cuda::PtrStep<float> conditionImg) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    if (conditionImgCopy.ptr(y)[x] < 10.f) {
                        return;
                    }
                    const int sizeSingle = kSize / 2;
                    float data[25];
                    for (int i = -sizeSingle; i < sizeSingle; i++) {
                        for (int j = -sizeSingle; j < sizeSingle; j++) {
                            if (x + j < 0 || x + j > cols - 1 || y + i < 0 || y + i > rows - 1) {
                                return;
                            }
                            if (conditionImgCopy.ptr(y + i)[x + j] < 5.f) {
                                conditionImg.ptr(y)[x] = 0;
                                return;
                            }
                            data[kSize * (i + sizeSingle) + (j + sizeSingle)] = src.ptr(y + i)[x + j];
                        }
                    }
                    sortData(data, kSize * kSize);
                    dst.ptr(y)[x] = data[(kSize * kSize) / 2];
                }
            }

            __device__ uchar lookGrayCodeTable(const uchar src) {
                switch (src) {
                    case 0:
                        return 0;
                    case 1:
                        return 1;
                    case 2:
                        return 2;
                    case 3:
                        return 3;
                    case 7:
                        return 4;
                    case 6:
                        return 5;
                    case 5:
                        return 6;
                    case 4:
                        return 7;
                    case 8:
                        return 8;
                    case 9:
                        return 9;
                    case 10:
                        return 10;
                    case 11:
                        return 11;
                    case 15:
                        return 12;
                    case 14:
                        return 13;
                    case 13:
                        return 14;
                    case 12:
                        return 15;
                }
            }

            //默认顺序：I2 -> I1 -> I3 -> I4 -> I2 -> I1 ...
            __global__ void prepare_FourFloorFourStep(
                    const cv::cuda::PtrStep<uchar> shift_0_,
                    const cv::cuda::PtrStep<uchar> shift_1_,
                    const cv::cuda::PtrStep<uchar> shift_2_,
                    const cv::cuda::PtrStep<uchar> shift_3_,
                    const cv::cuda::PtrStep<uchar> gray_0_,
                    const cv::cuda::PtrStep<uchar> gray_1_,
                    const int rows, const int cols,
                    cv::cuda::PtrStep<float> wrapImg, cv::cuda::PtrStep<float> conditionImg,
                    cv::cuda::PtrStep<float> kFloorImg_0_, cv::cuda::PtrStep<float> kFloorImg_1_,
                    const bool isOdd) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    float env_A = ((float) shift_1_.ptr(y)[x] + (float) shift_2_.ptr(y)[x]) / 2.f;
                    float tex_B = cuda::std::sqrtf(cuda::std::powf((float) shift_2_.ptr(y)[x] - env_A, 2) + cuda::std::powf((float) shift_3_.ptr(y)[x] - env_A, 2));

                    conditionImg.ptr(y)[x] = tex_B;

                    if (tex_B < 10.f)
                        return;

                    float wrapVal = 0;
                    if (isOdd)
                        wrapVal = cuda::std::atan2f((float) shift_3_.ptr(y)[x] - env_A, env_A - (float) shift_2_.ptr(y)[x]);
                    else
                        wrapVal = cuda::std::atan2f(env_A - (float) shift_2_.ptr(y)[x], (float) shift_3_.ptr(y)[x] - env_A);

                    kFloorImg_0_.ptr(y)[x] = ((float) gray_0_.ptr(y)[x] - env_A) / tex_B;
                    kFloorImg_1_.ptr(y)[x] = ((float) gray_1_.ptr(y)[x] - env_A) / tex_B;
                    wrapImg.ptr(y)[x] = wrapVal;

                    return;
                }
            }

            __global__ void caculateKFloorImg(
                    const cv::cuda::PtrStep<float> kFloorImg_0_,
                    const cv::cuda::PtrStep<float> kFloorImg_1_,
                    const cv::cuda::PtrStep<float> conditionImg,
                    const cv::cuda::PtrStep<float> threshodVal,
                    const int rows, const int cols, cv::cuda::PtrStep<uchar> kFloorImg) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    if (conditionImg.ptr(y)[x] < 10.f) {
                        return;
                    }

                    float grayVal_0_ = kFloorImg_0_.ptr(y)[x];
                    float grayVal_1_ = kFloorImg_1_.ptr(y)[x];
                    float minDistance = 5.f;
                    uchar currentFloor_0_ = 0;
                    uchar currentFloor_1_ = 0;
                    for (int i = 0; i < 4; i++) {
                        float distance = cuda::std::abs(grayVal_0_ - threshodVal.ptr(i)[0]);
                        if (distance < minDistance) {
                            minDistance = distance;
                            currentFloor_0_ = i;
                        }
                    }
                    minDistance = 5.f;
                    for (int i = 0; i < 4; i++) {
                        float distance = cuda::std::abs(grayVal_1_ - threshodVal.ptr(i)[0]);
                        if (distance < minDistance) {
                            minDistance = distance;
                            currentFloor_1_ = i;
                        }
                    }

                    kFloorImg.ptr(y)[x] = lookGrayCodeTable(currentFloor_0_ + 4 * currentFloor_1_);
                    return;
                }
            }

            __global__ void getUnwrapImg_FourFloorFourStep(
                    const cv::cuda::PtrStep<uchar> kFloorImg,
                    const cv::cuda::PtrStep<float> conditionImg,
                    const cv::cuda::PtrStep<float> wrapImg,
                    const int rows, const int cols,
                    cv::cuda::PtrStep<float> unwrapImg) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    if (conditionImg.ptr(y)[x] < 10.f) {
                        unwrapImg.ptr(y)[x] = -5.f;
                        return;
                    }

                    if (cuda::std::abs(wrapImg.ptr(y)[x]) < CV_PI / 3)
                        unwrapImg.ptr(y)[x] = wrapImg.ptr(y)[x] + CV_2PI * kFloorImg.ptr(y)[x];
                    else {
                        int minLocation = INT_MAX;
                        float minWrapVal = FLT_MAX;
                        const uchar index = kFloorImg.ptr(y)[x];
                        const float wrapVal = wrapImg.ptr(y)[x];
                        for (int k = -80; k < 80; k++) {
                            if (k + x < 0 || k + x > cols - 1)
                                continue;
                            uchar indexFind = kFloorImg.ptr(y)[k + x];
                            if (indexFind != index || conditionImg.ptr(y)[k + x] < 10.f)
                                continue;
                            float currentWrapVal = cuda::std::abs(wrapImg.ptr(y)[k + x]);
                            if (currentWrapVal < minWrapVal) {
                                minWrapVal = currentWrapVal;
                                minLocation = k + x;
                            }
                        }

                        if (x < minLocation) {
                            if (wrapVal > CV_PI / 3)
                                unwrapImg.ptr(y)[x] = wrapVal + CV_2PI * (index - 1);
                            else
                                unwrapImg.ptr(y)[x] = wrapVal + CV_2PI * index;
                        } else {
                            if (wrapVal < -CV_PI / 3)
                                unwrapImg.ptr(y)[x] = wrapVal + CV_2PI * (index + 1);
                            else
                                unwrapImg.ptr(y)[x] = wrapVal + CV_2PI * index;
                        }
                    }
                    return;
                }
            }

            __global__ void atan3M_ShiftGray(
                    const cv::cuda::PtrStep<uchar> shift_0_0_,
                    const cv::cuda::PtrStep<uchar> shift_0_1_,
                    const cv::cuda::PtrStep<uchar> shift_0_2_,
                    const cv::cuda::PtrStep<uchar> shift_1_0_,
                    const cv::cuda::PtrStep<uchar> shift_1_1_,
                    const cv::cuda::PtrStep<uchar> shift_1_2_,
                    const cv::cuda::PtrStep<uchar> gray_0_,
                    const cv::cuda::PtrStep<uchar> gray_1_,
                    const cv::cuda::PtrStep<uchar> gray_2_,
                    const cv::cuda::PtrStep<uchar> gray_3_,
                    const int rows, const int cols,
                    cv::cuda::PtrStep<float> wrapImg_1_, cv::cuda::PtrStep<float> conditionImg_1_,
                    cv::cuda::PtrStep<float> wrapImg_2_, cv::cuda::PtrStep<float> conditionImg_2_,
                    cv::cuda::PtrStep<uchar> floor_K) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    const uchar valShift_0_0_ = shift_0_0_.ptr(y)[x];
                    const uchar valShift_0_1_ = shift_0_1_.ptr(y)[x];
                    const uchar valShift_0_2_ = shift_0_2_.ptr(y)[x];
                    const uchar valShift_1_0_ = shift_1_0_.ptr(y)[x];
                    const uchar valShift_1_1_ = shift_1_1_.ptr(y)[x];
                    const uchar valShift_1_2_ = shift_1_2_.ptr(y)[x];
                    const uchar valGray_0_ = gray_0_.ptr(y)[x];
                    const uchar valGray_1_ = gray_1_.ptr(y)[x];
                    const uchar valGray_2_ = gray_2_.ptr(y)[x];
                    const uchar valGray_3_ = gray_3_.ptr(y)[x];
                    wrapImg_1_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_0_2_ - valShift_0_1_), 2.0f * valShift_0_0_ - valShift_0_2_ - valShift_0_1_);
                    wrapImg_2_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_1_2_ - valShift_1_1_), 2.0f * valShift_1_0_ - valShift_1_2_ - valShift_1_1_);
                    float averageImg = (valShift_1_0_ + valShift_1_1_ + valShift_1_2_) / 3.0f;
                    conditionImg_1_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_0_0_ - valShift_0_2_, 2) + std::powf(2.0f * valShift_0_1_ - valShift_0_0_ - valShift_0_2_, 2)) / 3.0f;
                    conditionImg_2_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_1_0_ - valShift_1_2_, 2) + std::powf(2.0f * valShift_1_1_ - valShift_1_0_ - valShift_1_2_, 2)) / 3.0f;
                    const uchar bool_0 = valGray_0_ > averageImg ? 1 : 0;
                    const uchar bool_1 = valGray_1_ > averageImg ? 1 : 0;
                    const uchar bool_2 = valGray_2_ > averageImg ? 1 : 0;
                    const uchar bool_3 = valGray_3_ > averageImg ? 1 : 0;
                    const uchar bit_3 = bool_0 ^ 0;
                    const uchar bit_2 = bool_1 ^ bit_3;
                    const uchar bit_1 = bool_2 ^ bit_2;
                    const uchar bit_0 = bool_3 ^ bit_1;
                    floor_K.ptr(y)[x] = bit_3 * 8 + bit_2 * 4 + bit_1 * 2 + bit_0;
                }
            }

            __global__ void atan3M_ShiftGrayFourFrame(
                    const cv::cuda::PtrStep<uchar> shift_0_0_,
                    const cv::cuda::PtrStep<uchar> shift_0_1_,
                    const cv::cuda::PtrStep<uchar> shift_0_2_,
                    const cv::cuda::PtrStep<uchar> shift_1_0_,
                    const cv::cuda::PtrStep<uchar> shift_1_1_,
                    const cv::cuda::PtrStep<uchar> shift_1_2_,
                    const cv::cuda::PtrStep<uchar> shift_2_0_,
                    const cv::cuda::PtrStep<uchar> shift_2_1_,
                    const cv::cuda::PtrStep<uchar> shift_2_2_,
                    const cv::cuda::PtrStep<uchar> shift_3_0_,
                    const cv::cuda::PtrStep<uchar> shift_3_1_,
                    const cv::cuda::PtrStep<uchar> shift_3_2_,
                    const cv::cuda::PtrStep<uchar> gray_0_, const cv::cuda::PtrStep<uchar> gray_1_,
                    const cv::cuda::PtrStep<uchar> gray_2_, const cv::cuda::PtrStep<uchar> gray_3_,
                    const int rows, const int cols,
                    cv::cuda::PtrStep<float> wrapImg_1_, cv::cuda::PtrStep<float> conditionImg_1_,
                    cv::cuda::PtrStep<float> wrapImg_2_, cv::cuda::PtrStep<float> conditionImg_2_,
                    cv::cuda::PtrStep<float> wrapImg_3_, cv::cuda::PtrStep<float> conditionImg_3_,
                    cv::cuda::PtrStep<float> wrapImg_4_, cv::cuda::PtrStep<float> conditionImg_4_,
                    cv::cuda::PtrStep<uchar> floor_K) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    const uchar valShift_0_0_ = shift_0_0_.ptr(y)[x];
                    const uchar valShift_0_1_ = shift_0_1_.ptr(y)[x];
                    const uchar valShift_0_2_ = shift_0_2_.ptr(y)[x];
                    const uchar valShift_1_0_ = shift_1_0_.ptr(y)[x];
                    const uchar valShift_1_1_ = shift_1_1_.ptr(y)[x];
                    const uchar valShift_1_2_ = shift_1_2_.ptr(y)[x];
                    const uchar valShift_2_0_ = shift_2_0_.ptr(y)[x];
                    const uchar valShift_2_1_ = shift_2_1_.ptr(y)[x];
                    const uchar valShift_2_2_ = shift_2_2_.ptr(y)[x];
                    const uchar valShift_3_0_ = shift_3_0_.ptr(y)[x];
                    const uchar valShift_3_1_ = shift_3_1_.ptr(y)[x];
                    const uchar valShift_3_2_ = shift_3_2_.ptr(y)[x];
                    const uchar valGray_0_ = gray_0_.ptr(y)[x];
                    const uchar valGray_1_ = gray_1_.ptr(y)[x];
                    const uchar valGray_2_ = gray_2_.ptr(y)[x];
                    const uchar valGray_3_ = gray_3_.ptr(y)[x];
                    wrapImg_1_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_0_2_ - valShift_0_1_), 2.0f * valShift_0_0_ - valShift_0_2_ - valShift_0_1_);
                    wrapImg_2_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_1_2_ - valShift_1_1_), 2.0f * valShift_1_0_ - valShift_1_2_ - valShift_1_1_);
                    wrapImg_3_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_2_2_ - valShift_2_1_), 2.0f * valShift_2_0_ - valShift_2_2_ - valShift_2_1_);
                    wrapImg_4_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_3_2_ - valShift_3_1_), 2.0f * valShift_3_0_ - valShift_3_2_ - valShift_3_1_);
                    float averageImg_1_ = (valShift_0_0_ + valShift_0_1_ + valShift_0_2_) / 3.0f;
                    float averageImg_2_ = (valShift_1_0_ + valShift_1_1_ + valShift_1_2_) / 3.0f;
                    float averageImg_3_ = (valShift_2_0_ + valShift_2_1_ + valShift_2_2_) / 3.0f;
                    float averageImg_4_ = (valShift_3_0_ + valShift_3_1_ + valShift_3_2_) / 3.0f;
                    conditionImg_1_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_0_0_ - valShift_0_2_, 2) + std::powf(2.0f * valShift_0_1_ - valShift_0_0_ - valShift_0_2_, 2)) / 3.0f;
                    conditionImg_2_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_1_0_ - valShift_1_2_, 2) + std::powf(2.0f * valShift_1_1_ - valShift_1_0_ - valShift_1_2_, 2)) / 3.0f;
                    conditionImg_3_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_2_0_ - valShift_2_2_, 2) + std::powf(2.0f * valShift_2_1_ - valShift_2_0_ - valShift_2_2_, 2)) / 3.0f;
                    conditionImg_4_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_3_0_ - valShift_3_2_, 2) + std::powf(2.0f * valShift_3_1_ - valShift_3_0_ - valShift_3_2_, 2)) / 3.0f;
                    const uchar bool_0 = valGray_0_ > averageImg_1_ ? 1 : 0;
                    const uchar bool_1 = valGray_1_ > averageImg_2_ ? 1 : 0;
                    const uchar bool_2 = valGray_2_ > averageImg_3_ ? 1 : 0;
                    const uchar bool_3 = valGray_3_ > averageImg_4_ ? 1 : 0;
                    const uchar bit_3 = bool_0 ^ 0;
                    const uchar bit_2 = bool_1 ^ bit_3;
                    const uchar bit_1 = bool_2 ^ bit_2;
                    const uchar bit_0 = bool_3 ^ bit_1;
                    floor_K.ptr(y)[x] = bit_3 * 8 + bit_2 * 4 + bit_1 * 2 + bit_0;
                }
            }

            __global__ void getUnwrapImg_ShiftGray(
                    const cv::cuda::PtrStep<float> absolutImgWhite,
                    const int rows, const int cols,
                    const cv::cuda::PtrStep<float> wrapImg_1_,
                    const cv::cuda::PtrStep<float> conditionImg_1_,
                    cv::cuda::PtrStep<float> unwrapImg_1_,
                    const cv::cuda::PtrStep<float> wrapImg_2_,
                    const cv::cuda::PtrStep<float> conditionImg_2_,
                    cv::cuda::PtrStep<float> unwrapImg_2_,
                    cv::cuda::PtrStep<uchar> floor_K) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                const float CV_2PI_DIV_3 = CV_2PI / 3;
                if (x < cols && y < rows) {
                    const int K = floor_K.ptr(y)[x];
                    if (conditionImg_1_.ptr(y)[x] < 10.f) {
                        unwrapImg_1_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_1_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -120; k < 120; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_1_.ptr(y)[index_search] < 10.f) {
                                continue;
                            }
                            float searchDisparityValue = cuda::std::abs(wrapImg_1_.ptr(y)[index_search] - CV_PI);
                            if (searchDisparityValue < minWrapValue &&
                                searchDisparityValue != 0) {
                                minWrapValue = searchDisparityValue;
                                refValue_1_compare = index_search;
                            }
                        }
                        if ((CV_2PI / 3 < wrapImg_1_.ptr(y)[x]) ||
                            (x < refValue_1_compare && cuda::std::abs(wrapImg_1_.ptr(y)[x]) < CV_2PI / 3)) {
                            unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * (K - 1);
                        } else {
                            unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * K;
                        }
                    }
                    if (conditionImg_2_.ptr(y)[x] < 10.0) {
                        unwrapImg_2_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_2_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -120; k < 120; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_2_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            float searchDisparityValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search] - CV_PI);
                            if (searchDisparityValue < minWrapValue && searchDisparityValue != 0) {
                                minWrapValue = searchDisparityValue;
                                refValue_2_compare = index_search;
                            }
                        }
                        if ((CV_2PI / 3 < wrapImg_2_.ptr(y)[x]) ||
                            (x < refValue_2_compare && cuda::std::abs(wrapImg_2_.ptr(y)[x]) < CV_2PI / 3)) {
                            unwrapImg_2_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * (K - 1);
                        } else {
                            unwrapImg_2_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * K;
                        }
                    }
                }
            }

            __global__ void getUnwrapImg_ShiftGrayFourFrame(
                    const cv::cuda::PtrStep<float> absolutImgWhite,
                    const int rows, const int cols,
                    const cv::cuda::PtrStep<float> wrapImg_1_,
                    const cv::cuda::PtrStep<float> conditionImg_1_,
                    cv::cuda::PtrStep<float> unwrapImg_1_,
                    const cv::cuda::PtrStep<float> wrapImg_2_,
                    const cv::cuda::PtrStep<float> conditionImg_2_,
                    cv::cuda::PtrStep<float> unwrapImg_2_,
                    const cv::cuda::PtrStep<float> wrapImg_3_,
                    const cv::cuda::PtrStep<float> conditionImg_3_,
                    cv::cuda::PtrStep<float> unwrapImg_3_,
                    const cv::cuda::PtrStep<float> wrapImg_4_,
                    const cv::cuda::PtrStep<float> conditionImg_4_,
                    cv::cuda::PtrStep<float> unwrapImg_4_,
                    cv::cuda::PtrStep<uchar> floor_K) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                const float CV_2PI_DIV_3 = CV_2PI / 3;
                if (x < cols && y < rows) {
                    const int K = floor_K.ptr(y)[x];
                    if (conditionImg_1_.ptr(y)[x] < 10.0) {
                        unwrapImg_1_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_1_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -50; k < 50; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_1_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            float searchDisparityValue = cuda::std::abs(wrapImg_1_.ptr(y)[index_search] - CV_PI);
                            if (searchDisparityValue < minWrapValue && searchDisparityValue != 0) {
                                minWrapValue = searchDisparityValue;
                                refValue_1_compare = index_search;
                            }
                        }
                        if ((CV_2PI / 3 < wrapImg_1_.ptr(y)[x]) ||
                            (x < refValue_1_compare && cuda::std::abs(wrapImg_1_.ptr(y)[x]) < CV_2PI / 3)) {
                            unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * (K - 1);
                        } else {
                            unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * K;
                        }
                    }
                    if (conditionImg_2_.ptr(y)[x] < 10.0) {
                        unwrapImg_2_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_2_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -50; k < 50; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K || conditionImg_2_.ptr(y)[index_search] < 5.0) {
                                continue;
                            }
                            float searchDisparityValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search] - CV_PI);
                            if (searchDisparityValue < minWrapValue && searchDisparityValue != 0) {
                                minWrapValue = searchDisparityValue;
                                refValue_2_compare = index_search;
                            }
                        }
                        if ((CV_2PI / 3 < wrapImg_2_.ptr(y)[x]) ||
                            (x < refValue_2_compare && cuda::std::abs(wrapImg_2_.ptr(y)[x]) < CV_2PI / 3)) {
                            unwrapImg_2_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * (K - 1);
                        } else {
                            unwrapImg_2_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * K;
                        }
                    }
                    if (conditionImg_3_.ptr(y)[x] < 10.0) {
                        unwrapImg_3_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_3_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -50; k < 50; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_3_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            float searchDisparityValue = cuda::std::abs(wrapImg_3_.ptr(y)[index_search] - CV_PI);
                            if (searchDisparityValue < minWrapValue && searchDisparityValue != 0) {
                                minWrapValue = searchDisparityValue;
                                refValue_3_compare = index_search;
                            }
                        }
                        if ((CV_2PI / 3 < wrapImg_3_.ptr(y)[x]) ||
                            (x < refValue_3_compare && cuda::std::abs(wrapImg_3_.ptr(y)[x]) < CV_2PI / 3)) {
                            unwrapImg_3_.ptr(y)[x] = wrapImg_3_.ptr(y)[x] + CV_2PI * (K - 1);
                        } else {
                            unwrapImg_3_.ptr(y)[x] = wrapImg_3_.ptr(y)[x] + CV_2PI * K;
                        }
                    }
                    if (conditionImg_4_.ptr(y)[x] < 10.0) {
                        unwrapImg_4_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_4_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -50; k < 50; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_4_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            float searchDisparityValue = cuda::std::abs(wrapImg_4_.ptr(y)[index_search] - CV_PI);
                            if (searchDisparityValue < minWrapValue && searchDisparityValue != 0) {
                                minWrapValue = searchDisparityValue;
                                refValue_4_compare = index_search;
                            }
                        }
                        if ((CV_2PI / 3 < wrapImg_4_.ptr(y)[x]) ||
                            (x < refValue_4_compare && cuda::std::abs(wrapImg_4_.ptr(y)[x]) < CV_2PI / 3)) {
                            unwrapImg_4_.ptr(y)[x] = wrapImg_4_.ptr(y)[x] + CV_2PI * (K - 1);
                        } else {
                            unwrapImg_4_.ptr(y)[x] = wrapImg_4_.ptr(y)[x] + CV_2PI * K;
                        }
                    }
                }
            }

            __global__ void atan3M_DevideSpace(
                    const cv::cuda::PtrStep<uchar> shift_0_0_,
                    const cv::cuda::PtrStep<uchar> shift_0_1_,
                    const cv::cuda::PtrStep<uchar> shift_0_2_,
                    const cv::cuda::PtrStep<uchar> gray_0_,
                    const cv::cuda::PtrStep<uchar> shift_1_0_,
                    const cv::cuda::PtrStep<uchar> shift_1_1_,
                    const cv::cuda::PtrStep<uchar> shift_1_2_,
                    const cv::cuda::PtrStep<uchar> gray_1_,
                    const cv::cuda::PtrStep<uchar> shift_2_0_,
                    const cv::cuda::PtrStep<uchar> shift_2_1_,
                    const cv::cuda::PtrStep<uchar> shift_2_2_,
                    const cv::cuda::PtrStep<uchar> gray_2_,
                    const cv::cuda::PtrStep<uchar> shift_3_0_,
                    const cv::cuda::PtrStep<uchar> shift_3_1_,
                    const cv::cuda::PtrStep<uchar> shift_3_2_,
                    const cv::cuda::PtrStep<uchar> gray_3_,
                    const int rows, const int cols,
                    cv::cuda::PtrStep<float> wrapImg1_1_, cv::cuda::PtrStep<float> wrapImg1_2_,
                    cv::cuda::PtrStep<float> wrapImg1_3_, cv::cuda::PtrStep<float> conditionImg_1_,
                    cv::cuda::PtrStep<float> wrapImg2_1_, cv::cuda::PtrStep<float> wrapImg2_2_,
                    cv::cuda::PtrStep<float> wrapImg2_3_, cv::cuda::PtrStep<float> conditionImg_2_,
                    cv::cuda::PtrStep<float> wrapImg3_1_, cv::cuda::PtrStep<float> wrapImg3_2_,
                    cv::cuda::PtrStep<float> wrapImg3_3_, cv::cuda::PtrStep<float> conditionImg_3_,
                    cv::cuda::PtrStep<float> wrapImg4_1_, cv::cuda::PtrStep<float> wrapImg4_2_,
                    cv::cuda::PtrStep<float> wrapImg4_3_, cv::cuda::PtrStep<float> conditionImg_4_,
                    cv::cuda::PtrStep<uchar> floor_K) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x < cols && y < rows) {
                    const uchar valShift_0_0_ = shift_0_0_.ptr(y)[x];
                    const uchar valShift_0_1_ = shift_0_1_.ptr(y)[x];
                    const uchar valShift_0_2_ = shift_0_2_.ptr(y)[x];
                    const uchar valGray_0_ = gray_0_.ptr(y)[x];
                    const uchar valShift_1_0_ = shift_1_0_.ptr(y)[x];
                    const uchar valShift_1_1_ = shift_1_1_.ptr(y)[x];
                    const uchar valShift_1_2_ = shift_1_2_.ptr(y)[x];
                    const uchar valGray_1_ = gray_1_.ptr(y)[x];
                    const uchar valShift_2_0_ = shift_2_0_.ptr(y)[x];
                    const uchar valShift_2_1_ = shift_2_1_.ptr(y)[x];
                    const uchar valShift_2_2_ = shift_2_2_.ptr(y)[x];
                    const uchar valGray_2_ = gray_2_.ptr(y)[x];
                    const uchar valShift_3_0_ = shift_3_0_.ptr(y)[x];
                    const uchar valShift_3_1_ = shift_3_1_.ptr(y)[x];
                    const uchar valShift_3_2_ = shift_3_2_.ptr(y)[x];
                    const uchar valGray_3_ = gray_3_.ptr(y)[x];
                    wrapImg1_1_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_0_2_ - valShift_0_1_), 2.0f * valShift_0_0_ - valShift_0_2_ - valShift_0_1_);
                    wrapImg1_2_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_0_0_ - valShift_0_2_), 2.0f * valShift_0_1_ - valShift_0_0_ - valShift_0_2_);
                    wrapImg1_3_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_0_1_ - valShift_0_0_), 2.0f * valShift_0_2_ - valShift_0_1_ - valShift_0_0_);
                    wrapImg2_1_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_1_2_ - valShift_1_1_), 2.0f * valShift_1_0_ - valShift_1_2_ - valShift_1_1_);
                    wrapImg2_2_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_1_0_ - valShift_1_2_), 2.0f * valShift_1_1_ - valShift_1_0_ - valShift_1_2_);
                    wrapImg2_3_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_1_1_ - valShift_1_0_), 2.0f * valShift_1_2_ - valShift_1_1_ - valShift_1_0_);
                    wrapImg3_1_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_2_2_ - valShift_2_1_), 2.0f * valShift_2_0_ - valShift_2_2_ - valShift_2_1_);
                    wrapImg3_2_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_2_0_ - valShift_2_2_), 2.0f * valShift_2_1_ - valShift_2_0_ - valShift_2_2_);
                    wrapImg3_3_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_2_1_ - valShift_2_0_), 2.0f * valShift_2_2_ - valShift_2_1_ - valShift_2_0_);
                    wrapImg4_1_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_3_2_ - valShift_3_1_), 2.0f * valShift_3_0_ - valShift_3_2_ - valShift_3_1_);
                    wrapImg4_2_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_3_0_ - valShift_3_2_), 2.0f * valShift_3_1_ - valShift_3_0_ - valShift_3_2_);
                    wrapImg4_3_.ptr(y)[x] = cuda::std::atan2f(cuda::std::sqrtf(3.0f) * (valShift_3_1_ - valShift_3_0_), 2.0f * valShift_3_2_ - valShift_3_1_ - valShift_3_0_);
                    float averageImg_1 = (valShift_0_0_ + valShift_0_1_ + valShift_0_2_) / 3.0f;
                    float averageImg_2 = (valShift_1_0_ + valShift_1_1_ + valShift_1_2_) / 3.0f;
                    float averageImg_3 = (valShift_2_0_ + valShift_2_1_ + valShift_2_2_) / 3.0f;
                    float averageImg_4 = (valShift_3_0_ + valShift_3_1_ + valShift_3_2_) / 3.0f;
                    conditionImg_1_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_0_0_ - valShift_0_2_, 2) + std::powf(2.0f * valShift_0_1_ - valShift_0_0_ - valShift_0_2_, 2)) / 3.0f;
                    conditionImg_2_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_1_0_ - valShift_1_2_, 2) + std::powf(2.0f * valShift_1_1_ - valShift_1_0_ - valShift_1_2_, 2)) / 3.0f;
                    conditionImg_3_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_2_0_ - valShift_2_2_, 2) + std::powf(2.0f * valShift_2_1_ - valShift_2_0_ - valShift_2_2_, 2)) / 3.0f;
                    conditionImg_4_.ptr(y)[x] = cuda::std::sqrtf(3.0f * cuda::std::powf(valShift_3_0_ - valShift_3_2_, 2) + std::powf(2.0f * valShift_3_1_ - valShift_3_0_ - valShift_3_2_, 2)) / 3.0f;
                    const uchar bool_0 = valGray_0_ > averageImg_1 ? 1 : 0;
                    const uchar bool_1 = valGray_1_ > averageImg_2 ? 1 : 0;
                    const uchar bool_2 = valGray_2_ > averageImg_3 ? 1 : 0;
                    const uchar bool_3 = valGray_3_ > averageImg_4 ? 1 : 0;
                    const uchar bit_3 = bool_0 ^ 0;
                    const uchar bit_2 = bool_1 ^ bit_3;
                    const uchar bit_1 = bool_2 ^ bit_2;
                    const uchar bit_0 = bool_3 ^ bit_1;
                    floor_K.ptr(y)[x] = bit_3 * 8 + bit_2 * 4 + bit_1 * 2 + bit_0;
                }
            }

            __global__ void getUnwrapImg_DevidedSpace(
                    const cv::cuda::PtrStep<float> absolutImgWhite,
                    const int rows, const int cols,
                    const cv::cuda::PtrStep<float> wrapImg_1_,
                    const cv::cuda::PtrStep<float> wrapImg_2_,
                    const cv::cuda::PtrStep<float> wrapImg_3_,
                    const cv::cuda::PtrStep<float> conditionImg_1_,
                    cv::cuda::PtrStep<float> unwrapImg_1_,
                    const cv::cuda::PtrStep<float> wrapImg_4_,
                    const cv::cuda::PtrStep<float> wrapImg_5_,
                    const cv::cuda::PtrStep<float> wrapImg_6_,
                    const cv::cuda::PtrStep<float> conditionImg_2_,
                    cv::cuda::PtrStep<float> unwrapImg_2_,
                    const cv::cuda::PtrStep<float> wrapImg_7_,
                    const cv::cuda::PtrStep<float> wrapImg_8_,
                    const cv::cuda::PtrStep<float> wrapImg_9_,
                    const cv::cuda::PtrStep<float> conditionImg_3_,
                    cv::cuda::PtrStep<float> unwrapImg_3_,
                    const cv::cuda::PtrStep<float> wrapImg_10_,
                    const cv::cuda::PtrStep<float> wrapImg_11_,
                    const cv::cuda::PtrStep<float> wrapImg_12_,
                    const cv::cuda::PtrStep<float> conditionImg_4_,
                    cv::cuda::PtrStep<float> unwrapImg_4_,
                    cv::cuda::PtrStep<uchar> floor_K) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                const float CV_2PI_DIV_3 = CV_2PI / 3;
                if (x < cols && y < rows) {
                    const float valAbs = absolutImgWhite.ptr(y)[x];
                    const int K = floor_K.ptr(y)[x];
                    if (valAbs <= 0) {
                        unwrapImg_1_.ptr(y)[x] = -5.f;
                        unwrapImg_2_.ptr(y)[x] = -5.f;
                        unwrapImg_3_.ptr(y)[x] = -5.f;
                        unwrapImg_4_.ptr(y)[x] = -5.f;
                        return;
                    }
                    const float refValue = valAbs - CV_2PI * K;
                    if (conditionImg_1_.ptr(y)[x] < 10.0) {
                        unwrapImg_1_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_1_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -120; k < 120; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_1_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            if (cuda::std::abs(wrapImg_2_.ptr(y)[index_search]) < minWrapValue &&
                                cuda::std::abs(wrapImg_2_.ptr(y)[index_search]) > 0) {
                                minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                                refValue_1_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * K;
                            }
                        }
                        if (cuda::std::abs(wrapImg_2_.ptr(y)[x]) < CV_PI / 3) {
                            unwrapImg_1_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * K;
                        } else if (refValue < refValue_1_compare) {
                            unwrapImg_1_.ptr(y)[x] = wrapImg_3_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                        } else {
                            unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                        }
                    }
                    if (conditionImg_2_.ptr(y)[x] < 10.0) {
                        unwrapImg_2_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_2_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -120; k < 120; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_2_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            if (cuda::std::abs(wrapImg_5_.ptr(y)[index_search]) < minWrapValue &&
                                cuda::std::abs(wrapImg_5_.ptr(y)[index_search]) > 0) {
                                minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                                refValue_2_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * floor_K.ptr(y)[index_search];
                            }
                        }
                        //使用的DLP3010计算出的包裹相位位移顺序相反
                        if (cuda::std::abs(wrapImg_5_.ptr(y)[x]) < CV_PI / 3) {
                            unwrapImg_2_.ptr(y)[x] = wrapImg_5_.ptr(y)[x] + CV_2PI * K;
                        } else if (refValue < refValue_2_compare) {
                            unwrapImg_2_.ptr(y)[x] = wrapImg_6_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                        } else {
                            unwrapImg_2_.ptr(y)[x] = wrapImg_4_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                        }
                    }
                    if (conditionImg_3_.ptr(y)[x] < 10.0) {
                        unwrapImg_3_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_3_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -120; k < 120; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 || floor_K.ptr(y)[index_search] != K || conditionImg_3_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            if (cuda::std::abs(wrapImg_8_.ptr(y)[index_search]) < minWrapValue && cuda::std::abs(wrapImg_8_.ptr(y)[index_search]) > 0) {
                                minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                                refValue_3_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * floor_K.ptr(y)[index_search];
                            }
                        }
                        if (cuda::std::abs(wrapImg_8_.ptr(y)[x]) < CV_PI / 3) {
                            unwrapImg_3_.ptr(y)[x] = wrapImg_8_.ptr(y)[x] + CV_2PI * K;
                        } else if (refValue < refValue_3_compare) {
                            unwrapImg_3_.ptr(y)[x] = wrapImg_9_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                        } else {
                            unwrapImg_3_.ptr(y)[x] = wrapImg_7_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                        }
                    }
                    if (conditionImg_4_.ptr(y)[x] < 10.0) {
                        unwrapImg_4_.ptr(y)[x] = -5.f;
                    } else {
                        float refValue_4_compare;
                        float minWrapValue = FLT_MAX;
                        for (int k = -120; k < 120; k++) {
                            int index_search = x + k;
                            if (index_search < 0 || index_search > cols - 1 ||
                                floor_K.ptr(y)[index_search] != K ||
                                conditionImg_4_.ptr(y)[index_search] < 10.0) {
                                continue;
                            }
                            if (cuda::std::abs(wrapImg_11_.ptr(y)[index_search]) < minWrapValue &&
                                cuda::std::abs(wrapImg_11_.ptr(y)[index_search]) > 0) {
                                minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                                refValue_4_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * floor_K.ptr(y)[index_search];
                            }
                        }
                        if (cuda::std::abs(wrapImg_11_.ptr(y)[x]) < CV_PI / 3) {
                            unwrapImg_4_.ptr(y)[x] = wrapImg_11_.ptr(y)[x] + CV_2PI * K;
                        } else if (refValue < refValue_4_compare) {
                            unwrapImg_4_.ptr(y)[x] = wrapImg_12_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                        } else {
                            unwrapImg_4_.ptr(y)[x] = wrapImg_10_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                        }
                    }
                }
            }

            __global__ void solvePhase_RefPlain(const cv::cuda::PtrStep<float> wrapImg, const cv::cuda::PtrStep<float> conditionImg,
                const cv::cuda::PtrStep<float> refPlainImg, const int rows, const int cols, 
                cv::cuda::PtrStep<float> unwrapImg, const bool isFarest) {
                const int x = blockDim.x * blockIdx.x + threadIdx.x;
                const int y = blockDim.y * blockIdx.y + threadIdx.y;
                if (x < cols && y < rows) {
                    if (conditionImg.ptr(y)[x] < 10.f) {
                        unwrapImg.ptr(y)[x] = -5.f;
                        return;
                    }
                    if (isFarest)
                        unwrapImg.ptr(y)[x] = wrapImg.ptr(y)[x] + cuda::std::floorf((refPlainImg.ptr(y)[x] - wrapImg.ptr(y)[x]) / CV_2PI) * CV_2PI;
                    else
                        unwrapImg.ptr(y)[x] = wrapImg.ptr(y)[x] + cuda::std::ceilf((refPlainImg.ptr(y)[x] - wrapImg.ptr(y)[x]) / CV_2PI) * CV_2PI;
                }
            }

            void solvePhasePrepare_DevidedSpace(
                    const cv::cuda::GpuMat &shift_0_0_,
                    const cv::cuda::GpuMat &shift_0_1_,
                    const cv::cuda::GpuMat &shift_0_2_,
                    const cv::cuda::GpuMat &gray_0_,
                    const cv::cuda::GpuMat &shift_1_0_,
                    const cv::cuda::GpuMat &shift_1_1_,
                    const cv::cuda::GpuMat &shift_1_2_,
                    const cv::cuda::GpuMat &gray_1_,
                    const cv::cuda::GpuMat &shift_2_0_,
                    const cv::cuda::GpuMat &shift_2_1_,
                    const cv::cuda::GpuMat &shift_2_2_,
                    const cv::cuda::GpuMat &gray_2_,
                    const cv::cuda::GpuMat &shift_3_0_,
                    const cv::cuda::GpuMat &shift_3_1_,
                    const cv::cuda::GpuMat &shift_3_2_,
                    const cv::cuda::GpuMat &gray_3_,
                    const int rows, const int cols,
                    cv::cuda::GpuMat &wrapImg1_1_, cv::cuda::GpuMat &wrapImg1_2_,
                    cv::cuda::GpuMat &wrapImg1_3_, cv::cuda::GpuMat &conditionImg_1_,
                    cv::cuda::GpuMat &wrapImg2_1_, cv::cuda::GpuMat &wrapImg2_2_,
                    cv::cuda::GpuMat &wrapImg2_3_, cv::cuda::GpuMat &conditionImg_2_,
                    cv::cuda::GpuMat &wrapImg3_1_, cv::cuda::GpuMat &wrapImg3_2_,
                    cv::cuda::GpuMat &wrapImg3_3_, cv::cuda::GpuMat &conditionImg_3_,
                    cv::cuda::GpuMat &wrapImg4_1_, cv::cuda::GpuMat &wrapImg4_2_,
                    cv::cuda::GpuMat &wrapImg4_3_, cv::cuda::GpuMat &conditionImg_4_,
                    cv::cuda::GpuMat &floor_K, const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                atan3M_DevideSpace<<<grid, block, 0, stream>>>(
                        shift_0_0_, shift_0_1_, shift_0_2_, gray_0_,
                        shift_1_0_, shift_1_1_, shift_1_2_, gray_1_,
                        shift_2_0_, shift_2_1_, shift_2_2_, gray_2_,
                        shift_3_0_, shift_3_1_, shift_3_2_, gray_3_,
                        rows, cols,
                        wrapImg1_1_, wrapImg1_2_, wrapImg1_3_, conditionImg_1_,
                        wrapImg2_1_, wrapImg2_2_, wrapImg2_3_, conditionImg_2_,
                        wrapImg3_1_, wrapImg3_2_, wrapImg3_3_, conditionImg_3_,
                        wrapImg4_1_, wrapImg4_2_, wrapImg4_3_, conditionImg_4_, floor_K);
            }

            void solvePhase_DevidedSpace(
                const cv::cuda::GpuMat &absolutImgWhite,
                const int rows, const int cols,
                const cv::cuda::GpuMat &wrapImg_1_0_, const cv::cuda::GpuMat &wrapImg_1_1_,
                const cv::cuda::GpuMat &wrapImg_1_2_, const cv::cuda::GpuMat &conditionImg_1_,
                cv::cuda::GpuMat &unwrapImg_1_,
                const cv::cuda::GpuMat &wrapImg_2_0_, const cv::cuda::GpuMat &wrapImg_2_1_,
                const cv::cuda::GpuMat &wrapImg_2_2_, const cv::cuda::GpuMat &conditionImg_2_,
                cv::cuda::GpuMat &unwrapImg_2_,
                const cv::cuda::GpuMat &wrapImg_3_0_, const cv::cuda::GpuMat &wrapImg_3_1_,
                const cv::cuda::GpuMat &wrapImg_3_2_, const cv::cuda::GpuMat &conditionImg_3_,
                cv::cuda::GpuMat &unwrapImg_3_,
                const cv::cuda::GpuMat &wrapImg_4_0_, const cv::cuda::GpuMat &wrapImg_4_1_,
                const cv::cuda::GpuMat &wrapImg_4_2_, const cv::cuda::GpuMat &conditionImg_4_,
                cv::cuda::GpuMat &unwrapImg_4_,
                cv::cuda::GpuMat &floor_K, const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                getUnwrapImg_DevidedSpace<<<grid, block, 0, stream>>>(
                        absolutImgWhite, rows, cols,
                        wrapImg_1_0_, wrapImg_1_1_, wrapImg_1_2_, conditionImg_1_, unwrapImg_1_,
                        wrapImg_2_0_, wrapImg_2_1_, wrapImg_2_2_, conditionImg_2_, unwrapImg_2_,
                        wrapImg_3_0_, wrapImg_3_1_, wrapImg_3_2_, conditionImg_3_, unwrapImg_3_,
                        wrapImg_4_0_, wrapImg_4_1_, wrapImg_4_2_, conditionImg_4_, unwrapImg_4_,
                        floor_K);
            }


            void solvePhasePrepare_ShiftGray(
                    const cv::cuda::GpuMat &shift_0_0_,
                    const cv::cuda::GpuMat &shift_0_1_,
                    const cv::cuda::GpuMat &shift_0_2_,
                    const cv::cuda::GpuMat &shift_1_0_,
                    const cv::cuda::GpuMat &shift_1_1_,
                    const cv::cuda::GpuMat &shift_1_2_,
                    const cv::cuda::GpuMat &gray_0_, const cv::cuda::GpuMat &gray_1_,
                    const cv::cuda::GpuMat &gray_2_, const cv::cuda::GpuMat &gray_3_,
                    const int rows, const int cols,
                    cv::cuda::GpuMat &wrapImg1, cv::cuda::GpuMat &conditionImg_1_,
                    cv::cuda::GpuMat &wrapImg2, cv::cuda::GpuMat &conditionImg_2_,
                    cv::cuda::GpuMat &floor_K, const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                atan3M_ShiftGray<<<grid, block, 0, stream>>>(shift_0_0_, shift_0_1_, shift_0_2_,
                                                             shift_1_0_, shift_1_1_, shift_1_2_,
                                                             gray_0_, gray_1_, gray_2_, gray_3_,
                                                             rows, cols,
                                                             wrapImg1, conditionImg_1_,
                                                             wrapImg2, conditionImg_2_, floor_K);
            }

            void solvePhasePrepare_ShiftGrayFourFrame(
                    const cv::cuda::GpuMat &shift_0_0_, const cv::cuda::GpuMat &shift_0_1_,
                    const cv::cuda::GpuMat &shift_0_2_,
                    const cv::cuda::GpuMat &shift_1_0_, const cv::cuda::GpuMat &shift_1_1_,
                    const cv::cuda::GpuMat &shift_1_2_,
                    const cv::cuda::GpuMat &shift_2_0_, const cv::cuda::GpuMat &shift_2_1_,
                    const cv::cuda::GpuMat &shift_2_2_,
                    const cv::cuda::GpuMat &shift_3_0_, const cv::cuda::GpuMat &shift_3_1_,
                    const cv::cuda::GpuMat &shift_3_2_,
                    const cv::cuda::GpuMat &gray_0_, const cv::cuda::GpuMat &gray_1_,
                    const cv::cuda::GpuMat &gray_2_, const cv::cuda::GpuMat &gray_3_,
                    const int rows, const int cols,
                    cv::cuda::GpuMat &wrapImg1, cv::cuda::GpuMat &conditionImg_1_,
                    cv::cuda::GpuMat &wrapImg2, cv::cuda::GpuMat &conditionImg_2_,
                    cv::cuda::GpuMat &wrapImg3, cv::cuda::GpuMat &conditionImg_3_,
                    cv::cuda::GpuMat &wrapImg4, cv::cuda::GpuMat &conditionImg_4_,
                    cv::cuda::GpuMat &floor_K, const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                atan3M_ShiftGrayFourFrame<<<grid, block, 0, stream>>>(
                        shift_0_0_, shift_0_1_, shift_0_2_,
                        shift_1_0_, shift_1_1_, shift_1_2_,
                        shift_2_0_, shift_2_1_, shift_2_2_,
                        shift_3_0_, shift_3_1_, shift_3_2_,
                        gray_0_, gray_1_, gray_2_, gray_3_,
                        rows, cols,
                        wrapImg1, conditionImg_1_,
                        wrapImg2, conditionImg_2_,
                        wrapImg3, conditionImg_3_,
                        wrapImg4, conditionImg_4_,
                        floor_K);
            }

            /** \AbsolutImgWhite **/
            void solvePhase_ShiftGray(
                    const cv::cuda::GpuMat &absolutImgWhite,
                    const int rows, const int cols,
                    const cv::cuda::GpuMat &wrapImg_1_, const cv::cuda::GpuMat &conditionImg_1_,
                    cv::cuda::GpuMat &unwrapImg_1_,
                    const cv::cuda::GpuMat &wrapImg_2_, const cv::cuda::GpuMat &conditionImg_2_,
                    cv::cuda::GpuMat &unwrapImg_2_,
                    cv::cuda::GpuMat &floor_K, const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                getUnwrapImg_ShiftGray<<<grid, block, 0, stream>>>(absolutImgWhite, rows, cols,
                                                                   wrapImg_1_, conditionImg_1_, unwrapImg_1_,
                                                                   wrapImg_2_, conditionImg_2_, unwrapImg_2_, floor_K);
                //cudaDeviceSynchronize();
                /*
        cv::Mat test1(800, 1280, CV_32FC1);
        cv::Mat test2(800, 1280, CV_32FC1);
        cv::Mat test3(800, 1280, CV_32FC1);
        cv::Mat test4(800, 1280, CV_32FC1);
        cv::Mat test5(800, 1280, CV_32FC1);
        cudaMemcpy(test1.data, absolutePhaseImg, size_molloc, cudaMemcpyDeviceToHost);
        cudaMemcpy(test2.data, &absolutePhaseImg[rows * cols], size_molloc, cudaMemcpyDeviceToHost);
        cudaMemcpy(test3.data, &absolutePhaseImg[2 * rows * cols], size_molloc, cudaMemcpyDeviceToHost);
        cudaMemcpy(test4.data, &absolutePhaseImg[3 * rows * cols], size_molloc, cudaMemcpyDeviceToHost);
        cudaMemcpy(test5.data, lower_high, size_molloc, cudaMemcpyDeviceToHost);
        */
            }

            void solvePhase_ShiftGrayFourFrame(
                    const cv::cuda::GpuMat &absolutImgWhite,
                    const int rows, const int cols,
                    const cv::cuda::GpuMat &wrapImg_1_, const cv::cuda::GpuMat &conditionImg_1_,
                    cv::cuda::GpuMat &unwrapImg_1_,
                    const cv::cuda::GpuMat &wrapImg_2_, const cv::cuda::GpuMat &conditionImg_2_,
                    cv::cuda::GpuMat &unwrapImg_2_,
                    const cv::cuda::GpuMat &wrapImg_3_, const cv::cuda::GpuMat &conditionImg_3_,
                    cv::cuda::GpuMat &unwrapImg_3_,
                    const cv::cuda::GpuMat &wrapImg_4_, const cv::cuda::GpuMat &conditionImg_4_,
                    cv::cuda::GpuMat &unwrapImg_4_,
                    cv::cuda::GpuMat &floor_K, const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                getUnwrapImg_ShiftGrayFourFrame<<<grid, block, 0, stream>>>(
                        absolutImgWhite, rows, cols,
                        wrapImg_1_, conditionImg_1_, unwrapImg_1_,
                        wrapImg_2_, conditionImg_2_, unwrapImg_2_,
                        wrapImg_3_, conditionImg_3_, unwrapImg_3_,
                        wrapImg_4_, conditionImg_4_, unwrapImg_4_,
                        floor_K);
            }

            void solvePhasePrepare_FourStepSixGray(
                    const cv::cuda::GpuMat &shift_0_, const cv::cuda::GpuMat &shift_1_,
                    const cv::cuda::GpuMat &shift_2_, const cv::cuda::GpuMat &shift_3_,
                    const int rows, const int cols, cv::cuda::GpuMat &wrapImg,
                    cv::cuda::GpuMat &averageImg, cv::cuda::GpuMat &conditionImg,
                    const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                atan2M_FourStepSixGray<<<grid, block, 0, stream>>>(
                        shift_0_, shift_1_, shift_2_, shift_3_,
                        rows, cols,
                        wrapImg, averageImg, conditionImg);
            }

            void solvePhasePrepare_FourFloorFourStep(
                    const cv::cuda::GpuMat &shift_0_, const cv::cuda::GpuMat &shift_1_,
                    const cv::cuda::GpuMat &shift_2_, const cv::cuda::GpuMat &shift_3_,
                    const cv::cuda::GpuMat &gray_0_, const cv::cuda::GpuMat &gray_1_,
                    cv::cuda::GpuMat &threshodVal, cv::cuda::GpuMat &threshodAdd,
                    cv::cuda::GpuMat &count,
                    const int rows, const int cols, cv::cuda::GpuMat &wrapImg,
                    cv::cuda::GpuMat &conditionImg, cv::cuda::GpuMat &medianFilter_0_,
                    cv::cuda::GpuMat &medianFilter_1_, cv::cuda::GpuMat &kFloorImg_0_,
                    cv::cuda::GpuMat &kFloorImg_1_, cv::cuda::GpuMat &kFloorImg,
                    const bool isOdd, const bool isFirstStart,
                    const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                const int countKmeans = 2;
                cv::cuda::GpuMat conditionImgCopy;

                prepare_FourFloorFourStep<<<grid, block, 0, stream>>>(
                        shift_0_, shift_1_, shift_2_, shift_3_,
                        gray_0_, gray_1_,
                        rows, cols,
                        wrapImg, conditionImg, kFloorImg_0_, kFloorImg_1_, isOdd);
                /*
        if(isFirstStart){
            cv::cuda::bilateralFilter(kFloorImg_0_, kFloorImg_0_, 5, 10, 10, 4, cvStream);
            cv::cuda::bilateralFilter(kFloorImg_1_, kFloorImg_1_, 5, 10, 10, 4, cvStream);
        }
        else if(isOdd)
            cv::cuda::bilateralFilter(kFloorImg_0_, kFloorImg_0_, 5, 10, 10, 4, cvStream);
        else
            cv::cuda::bilateralFilter(kFloorImg_1_, kFloorImg_1_, 5, 10, 10, 4, cvStream);
        */

                conditionImg.copyTo(conditionImgCopy, cvStream);

                if (isFirstStart) {
                    medianFilter<<<grid, block, 0, stream>>>(
                            kFloorImg_0_, conditionImgCopy,
                            5, rows, cols, medianFilter_0_, conditionImg);
                    medianFilter<<<grid, block, 0, stream>>>(
                            kFloorImg_1_, conditionImgCopy,
                            5, rows, cols, medianFilter_1_, conditionImg);
                } else if (isOdd)
                    medianFilter<<<grid, block, 0, stream>>>(
                            kFloorImg_0_, conditionImgCopy,
                            5, rows, cols, medianFilter_0_, conditionImg);
                else
                    medianFilter<<<grid, block, 0, stream>>>(
                            kFloorImg_1_, conditionImgCopy,
                            5, rows, cols, medianFilter_1_, conditionImg);

                for (int i = 0; i < countKmeans; i++)
                    kMeansCluster<<<grid, block, 0, stream>>>(
                            kFloorImg_0_, conditionImg,
                            rows, cols, threshodVal, threshodAdd, count);
                /*
        caculateKFloorImg << <grid, block, 0, stream >> > (medianFilter_0_, medianFilter_1_, conditionImg,
            centroid_dark, centroid_lightDark, centroid_lightWhite, centroid_white,
            rows, cols, kFloorImg);
        */
                caculateKFloorImg<<<grid, block, 0, stream>>>(
                        medianFilter_0_, medianFilter_1_, conditionImg,
                        threshodVal, rows, cols, kFloorImg);
            }

            void solvePhase_FourFloorFourStep(
                    const cv::cuda::GpuMat &kFloorImg, const cv::cuda::GpuMat &conditionImg,
                    const cv::cuda::GpuMat &wrapImg,
                    const int rows, const int cols, cv::cuda::GpuMat &unwrapImg,
                    const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                getUnwrapImg_FourFloorFourStep<<<grid, block, 0, stream>>>(
                        kFloorImg, conditionImg, wrapImg,
                        rows, cols, unwrapImg);
            }

            void solvePhase_FourStepSixGray(
                const cv::cuda::GpuMat &Gray_0_, const cv::cuda::GpuMat &Gray_1_,
                const cv::cuda::GpuMat &Gray_2_, const cv::cuda::GpuMat &Gray_3_,
                const cv::cuda::GpuMat &Gray_4_, const cv::cuda::GpuMat &Gray_5_,
                const int rows, const int cols,
                const cv::cuda::GpuMat &averageImg, const cv::cuda::GpuMat &conditionImg,
                const cv::cuda::GpuMat &wrapImg, cv::cuda::GpuMat &unwrapImg,
                const dim3 block, cv::cuda::Stream &cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                getUnwrapImg_FourStepSixGray<<<grid, block, 0, stream>>>(
                    Gray_0_, Gray_1_, Gray_2_, Gray_3_, Gray_4_, Gray_5_,
                    rows, cols,
                    averageImg, conditionImg, wrapImg, unwrapImg);
            }

            void refPlainSolvePhase(const cv::cuda::GpuMat& wrapImg, const cv::cuda::GpuMat& conditionImg, 
                const cv::cuda::GpuMat& refPlainImg, const int rows, const int cols,
                cv::cuda::GpuMat &unwrapImg, const bool isFarest,
                const dim3 block, cv::cuda::Stream& cvStream) {
                cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
                dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
                solvePhase_RefPlain<<<grid, block, 0, stream>>>(wrapImg, conditionImg,
                                    refPlainImg, rows, cols,
                                    unwrapImg, isFarest);
            }
        }// namespace cudaFunc
    }// namespace phaseSolver
}// namespace sl