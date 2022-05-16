#include <Restructor/cuda/include/cudaTypeDef.cuh>

namespace PhaseSolverType {
    namespace cudaFunc {
        __global__ void atan2M_FourStepSixGray(const cv::cuda::PtrStep<uchar> shift_0_, const cv::cuda::PtrStep<uchar> shift_1_, const cv::cuda::PtrStep<uchar> shift_2_, const cv::cuda::PtrStep<uchar> shift_3_,
                                               const int rows, const int cols, cv::cuda::PtrStep<float> wrapImg, cv::cuda::PtrStep<float> averageImg, cv::cuda::PtrStep<float> conditionImg) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x < cols && y < rows) {
                wrapImg.ptr(y)[x] = cuda::std::atan2f(shift_3_.ptr(y)[x] - shift_1_.ptr(y)[x] , shift_0_.ptr(y)[x] - shift_2_.ptr(y)[x]);
                averageImg.ptr(y)[x] = (shift_0_.ptr(y)[x] + shift_1_.ptr(y)[x] + shift_2_.ptr(y)[x] + shift_3_.ptr(y)[x]) / 4.0f;
                conditionImg.ptr(y)[x] = cuda::std::sqrt((cuda::std::pow((shift_3_.ptr(y)[x] - shift_1_.ptr(y)[x]), 2) + cuda::std::pow((shift_0_.ptr(y)[x] - shift_2_.ptr(y)[x]), 2))) / 2.0f;
            }
        }

        __global__ void getUnwrapImg_FourStepSixGray(const cv::cuda::PtrStep<uchar> Gray_0_, const cv::cuda::PtrStep<uchar> Gray_1_, const cv::cuda::PtrStep<uchar> Gray_2_, const cv::cuda::PtrStep<uchar> Gray_3_, const cv::cuda::PtrStep<uchar> Gray_4_, const cv::cuda::PtrStep<uchar> Gray_5_,
                                                     const int rows, const int cols, const cv::cuda::PtrStep<float> averageImg, const cv::cuda::PtrStep<float> conditionImg, const cv::cuda::PtrStep<float> wrapImg, cv::cuda::PtrStep<float> unwrapImg) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x < cols && y < rows) {
                if (conditionImg.ptr(y)[x] < 20.0) {
                    unwrapImg.ptr(y)[x] = 0;
                }
                else {
                    const uchar bool_0 = Gray_0_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                    const uchar bool_1 = Gray_1_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                    const uchar bool_2 = Gray_2_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                    const uchar bool_3 = Gray_3_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                    const uchar bool_4 = Gray_4_.ptr(y)[x] > averageImg.ptr(y)[x] ? 1 : 0;
                    const uchar bool_5 = Gray_5_.ptr(y)[x] > averageImg.ptr(y)[x]? 1 : 0;
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

        __global__ void atan3M_ShiftGray(const cv::cuda::PtrStep<uchar> shift_0_0_, const cv::cuda::PtrStep<uchar> shift_0_1_, const cv::cuda::PtrStep<uchar> shift_0_2_,
            const cv::cuda::PtrStep<uchar> shift_1_0_, const cv::cuda::PtrStep<uchar> shift_1_1_, const cv::cuda::PtrStep<uchar> shift_1_2_,
            const cv::cuda::PtrStep<uchar> gray_0_, const cv::cuda::PtrStep<uchar> gray_1_, const cv::cuda::PtrStep<uchar> gray_2_, const cv::cuda::PtrStep<uchar> gray_3_,
            const int rows, const int cols,
            cv::cuda::PtrStep<float> wrapImg_1_, cv::cuda::PtrStep<float> conditionImg_1_,
            cv::cuda::PtrStep<float> wrapImg_2_, cv::cuda::PtrStep<float> conditionImg_2_, cv::cuda::PtrStep<uchar> floor_K) {
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
                floor_K.ptr(y)[x] = 16 - (bit_3 * 8 + bit_2 * 4 + bit_1 * 2 + bit_0);
            }
        }

        __global__ void getUnwrapImg_ShiftGray(const cv::cuda::PtrStep<float> absolutImgWhite, const int rows, const int cols,
            const cv::cuda::PtrStep<float> wrapImg_1_, const cv::cuda::PtrStep<float> conditionImg_1_, cv::cuda::PtrStep<float> unwrapImg_1_,
            const cv::cuda::PtrStep<float> wrapImg_2_, const cv::cuda::PtrStep<float> conditionImg_2_, cv::cuda::PtrStep<float> unwrapImg_2_, cv::cuda::PtrStep<uchar> floor_K) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const float CV_2PI_DIV_3 = CV_2PI / 3;
            if (x < cols && y < rows) {
                /*
                float absolutImgWhite_ = absolutImgWhite.ptr(y)[x];
                if (absolutImgWhite_ <= 0) {
                    unwrapImg_1_.ptr(y)[x] = 0;
                    unwrapImg_2_.ptr(y)[x] = 0;
                    return;
                }*/
                const int K = floor_K.ptr(y)[x];
                if (conditionImg_1_.ptr(y)[x] < 20.0) {
                    unwrapImg_1_.ptr(y)[x] = 0;
                }
                else {
                    float refValue_1_compare;
                    float minWrapValue = FLT_MAX;
                    for (int k = -400; k < 400; k++) {
                        int index_search = x + k;
                        if (index_search<0 || index_search >cols - 1 || floor_K.ptr(y)[index_search] != K || conditionImg_1_.ptr(y)[index_search] < 5.0) {
                            continue;
                        }
                        float searchDisparityValue = cuda::std::abs(wrapImg_1_.ptr(y)[index_search] - CV_PI);
                        if (searchDisparityValue < minWrapValue && searchDisparityValue != 0) {
                            minWrapValue = searchDisparityValue;
                            refValue_1_compare = index_search;
                        }
                    }
                    if ( (CV_2PI / 3 <= wrapImg_1_.ptr(y)[x]) || (x < refValue_1_compare && cuda::std::abs(wrapImg_1_.ptr(y)[x]) < CV_2PI/3)) {
                        unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * (K - 1);
                        //lower_high[index] = 0;
                    }
                    else {
                        unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * K;
                        //lower_high[index] = 128;
                    }
                }
                if (conditionImg_2_.ptr(y)[x] < 20.0) {
                    unwrapImg_2_.ptr(y)[x] = 0;
                }
                else {
                    float refValue_2_compare;
                    float minWrapValue = FLT_MAX;
                    for (int k = -400; k < 400; k++) {
                        int index_search = x + k;
                        if (index_search<0 || index_search >cols - 1 || floor_K.ptr(y)[index_search] != K || conditionImg_2_.ptr(y)[index_search] < 5.0) {
                            continue;
                        }
                        float searchDisparityValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search] - CV_PI);
                        if (searchDisparityValue < minWrapValue && searchDisparityValue != 0) {
                            minWrapValue = searchDisparityValue;
                            refValue_2_compare = index_search;
                        }
                    }
                    if ((CV_2PI / 3 <= wrapImg_2_.ptr(y)[x]) || (x < refValue_2_compare && cuda::std::abs(wrapImg_2_.ptr(y)[x]) < CV_2PI / 3)) {
                        unwrapImg_2_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * (K - 1);
                        //lower_high[index] = 0;
                    }
                    else {
                        unwrapImg_2_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * K;
                        //lower_high[index] = 128;
                    }
                }
            }
        }

        __global__ void atan3M_DevideSpace(const cv::cuda::PtrStep<uchar> shift_0_0_, const cv::cuda::PtrStep<uchar> shift_0_1_, const cv::cuda::PtrStep<uchar> shift_0_2_, const cv::cuda::PtrStep<uchar> gray_0_,
            const cv::cuda::PtrStep<uchar> shift_1_0_, const cv::cuda::PtrStep<uchar> shift_1_1_, const cv::cuda::PtrStep<uchar> shift_1_2_, const cv::cuda::PtrStep<uchar> gray_1_,
            const cv::cuda::PtrStep<uchar> shift_2_0_, const cv::cuda::PtrStep<uchar> shift_2_1_, const cv::cuda::PtrStep<uchar> shift_2_2_, const cv::cuda::PtrStep<uchar> gray_2_,
            const cv::cuda::PtrStep<uchar> shift_3_0_, const cv::cuda::PtrStep<uchar> shift_3_1_, const cv::cuda::PtrStep<uchar> shift_3_2_, const cv::cuda::PtrStep<uchar> gray_3_,
            const int rows, const int cols,
            cv::cuda::PtrStep<float> wrapImg1_1_, cv::cuda::PtrStep<float> wrapImg1_2_, cv::cuda::PtrStep<float> wrapImg1_3_, cv::cuda::PtrStep<float> conditionImg_1_,
            cv::cuda::PtrStep<float> wrapImg2_1_, cv::cuda::PtrStep<float> wrapImg2_2_, cv::cuda::PtrStep<float> wrapImg2_3_, cv::cuda::PtrStep<float> conditionImg_2_,
            cv::cuda::PtrStep<float> wrapImg3_1_, cv::cuda::PtrStep<float> wrapImg3_2_, cv::cuda::PtrStep<float> wrapImg3_3_, cv::cuda::PtrStep<float> conditionImg_3_,
            cv::cuda::PtrStep<float> wrapImg4_1_, cv::cuda::PtrStep<float> wrapImg4_2_, cv::cuda::PtrStep<float> wrapImg4_3_, cv::cuda::PtrStep<float> conditionImg_4_, cv::cuda::PtrStep<uchar> floor_K) {
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

        __global__ void getUnwrapImg_DevidedSpace(const cv::cuda::PtrStep<float> absolutImgWhite, const int rows, const int cols,
            const cv::cuda::PtrStep<float> wrapImg_1_, const cv::cuda::PtrStep<float> wrapImg_2_, const cv::cuda::PtrStep<float> wrapImg_3_, const cv::cuda::PtrStep<float> conditionImg_1_, cv::cuda::PtrStep<float> unwrapImg_1_,
            const cv::cuda::PtrStep<float> wrapImg_4_, const cv::cuda::PtrStep<float> wrapImg_5_, const cv::cuda::PtrStep<float> wrapImg_6_, const cv::cuda::PtrStep<float> conditionImg_2_, cv::cuda::PtrStep<float> unwrapImg_2_,
            const cv::cuda::PtrStep<float> wrapImg_7_, const cv::cuda::PtrStep<float> wrapImg_8_, const cv::cuda::PtrStep<float> wrapImg_9_, const cv::cuda::PtrStep<float> conditionImg_3_, cv::cuda::PtrStep<float> unwrapImg_3_,
            const cv::cuda::PtrStep<float> wrapImg_10_, const cv::cuda::PtrStep<float> wrapImg_11_, const cv::cuda::PtrStep<float> wrapImg_12_, const cv::cuda::PtrStep<float> conditionImg_4_, cv::cuda::PtrStep<float> unwrapImg_4_, cv::cuda::PtrStep<uchar> floor_K) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const float CV_2PI_DIV_3 = CV_2PI / 3;
            if (x < cols && y < rows) {
                const float valAbs = absolutImgWhite.ptr(y)[x];
                const int K = floor_K.ptr(y)[x];
                if (valAbs <= 0) {
                    unwrapImg_1_.ptr(y)[x] = 0;
                    unwrapImg_2_.ptr(y)[x] = 0;
                    unwrapImg_3_.ptr(y)[x] = 0;
                    unwrapImg_4_.ptr(y)[x] = 0;
                    return;
                }
                const float refValue = valAbs - CV_2PI * K;
                if (conditionImg_1_.ptr(y)[x] < 20.0) {
                    unwrapImg_1_.ptr(y)[x] = 0;
                }
                else {
                    float refValue_1_compare;
                    float minWrapValue = FLT_MAX;
                    for (int k = -400; k < 400; k++) {
                        int index_search = x + k;
                        if (index_search<0 || index_search >cols - 1 || floor_K.ptr(y)[index_search] != K || conditionImg_1_.ptr(y)[index_search] < 5.0) {
                            continue;
                        }
                        if (cuda::std::abs(wrapImg_2_.ptr(y)[index_search]) < minWrapValue && cuda::std::abs(wrapImg_2_.ptr(y)[index_search]) > 0) {
                            minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                            refValue_1_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * K;
                        }
                    }
                    if (cuda::std::abs(wrapImg_2_.ptr(y)[x]) < CV_PI / 3) {
                        unwrapImg_1_.ptr(y)[x] = wrapImg_2_.ptr(y)[x] + CV_2PI * K;
                    }
                    else if (refValue < refValue_1_compare) {
                        unwrapImg_1_.ptr(y)[x] = wrapImg_3_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                    }
                    else {
                        unwrapImg_1_.ptr(y)[x] = wrapImg_1_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                    }
                }
                if (conditionImg_2_.ptr(y)[x] < 20.0) {
                    unwrapImg_2_.ptr(y)[x] = 0;
                }
                else {
                    float refValue_2_compare;
                    float minWrapValue = FLT_MAX;
                    for (int k = -400; k < 400; k++) {
                        int index_search = x + k;
                        if (index_search<0 || index_search >cols - 1 || floor_K.ptr(y)[index_search] != K || conditionImg_2_.ptr(y)[index_search] < 5.0) {
                            continue;
                        }
                        if (cuda::std::abs(wrapImg_5_.ptr(y)[index_search]) < minWrapValue && cuda::std::abs(wrapImg_5_.ptr(y)[index_search]) > 0) {
                            minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                            refValue_2_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * floor_K.ptr(y)[index_search];
                        }
                    }
                    //使用的DLP3010计算出的包裹相位位移顺序相反
                    if (cuda::std::abs(wrapImg_5_.ptr(y)[x]) < CV_PI / 3) {
                        unwrapImg_2_.ptr(y)[x] = wrapImg_5_.ptr(y)[x] + CV_2PI * K;
                    }
                    else if (refValue < refValue_2_compare) {
                        unwrapImg_2_.ptr(y)[x] = wrapImg_6_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                    }
                    else {
                        unwrapImg_2_.ptr(y)[x] = wrapImg_4_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                    }

                }
                if (conditionImg_3_.ptr(y)[x] < 20.0) {
                    unwrapImg_3_.ptr(y)[x] = 0;
                }
                else {
                    float refValue_3_compare;
                    float minWrapValue = FLT_MAX;
                    for (int k = -400; k < 400; k++) {
                        int index_search = x + k;
                        if (index_search<0 || index_search >cols - 1 || floor_K.ptr(y)[index_search] != K || conditionImg_3_.ptr(y)[index_search] < 5.0) {
                            continue;
                        }
                        if (cuda::std::abs(wrapImg_8_.ptr(y)[index_search]) < minWrapValue && cuda::std::abs(wrapImg_8_.ptr(y)[index_search]) > 0) {
                            minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                            refValue_3_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * floor_K.ptr(y)[index_search];
                        }
                    }
                    if (cuda::std::abs(wrapImg_8_.ptr(y)[x]) < CV_PI / 3) {
                        unwrapImg_3_.ptr(y)[x] = wrapImg_8_.ptr(y)[x] + CV_2PI * K;
                    }
                    else if (refValue < refValue_3_compare) {
                        unwrapImg_3_.ptr(y)[x] = wrapImg_9_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                    }
                    else {
                        unwrapImg_3_.ptr(y)[x] = wrapImg_7_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                    }
                }
                if (conditionImg_4_.ptr(y)[x] < 20.0) {
                    unwrapImg_4_.ptr(y)[x] = 0;
                }
                else {
                    float refValue_4_compare;
                    float minWrapValue = FLT_MAX;
                    for (int k = -400; k < 400; k++) {
                        int index_search = x + k;
                        if (index_search<0 || index_search >cols - 1 || floor_K.ptr(y)[index_search] != K || conditionImg_4_.ptr(y)[index_search] < 5.0) {
                            continue;
                        }
                        if (cuda::std::abs(wrapImg_11_.ptr(y)[index_search]) < minWrapValue && cuda::std::abs(wrapImg_11_.ptr(y)[index_search]) > 0) {
                            minWrapValue = cuda::std::abs(wrapImg_2_.ptr(y)[index_search]);
                            refValue_4_compare = absolutImgWhite.ptr(y)[index_search] - CV_2PI * floor_K.ptr(y)[index_search];
                        }
                    }
                    if (cuda::std::abs(wrapImg_11_.ptr(y)[x]) < CV_PI / 3) {
                        unwrapImg_4_.ptr(y)[x] = wrapImg_11_.ptr(y)[x] + CV_2PI * K;
                    }
                    else if (refValue < refValue_4_compare) {
                        unwrapImg_4_.ptr(y)[x] = wrapImg_12_.ptr(y)[x] + CV_2PI * K - CV_2PI_DIV_3;
                    }
                    else {
                        unwrapImg_4_.ptr(y)[x] = wrapImg_10_.ptr(y)[x] + CV_2PI * K + CV_2PI_DIV_3;
                    }
                }
            }
        }

        void solvePhasePrepare_DevidedSpace(const cv::cuda::GpuMat& shift_0_0_, const cv::cuda::GpuMat & shift_0_1_, const cv::cuda::GpuMat & shift_0_2_, const cv::cuda::GpuMat & gray_0_,
            const cv::cuda::GpuMat& shift_1_0_, const cv::cuda::GpuMat & shift_1_1_, const cv::cuda::GpuMat & shift_1_2_, const cv::cuda::GpuMat & gray_1_,
            const cv::cuda::GpuMat& shift_2_0_, const cv::cuda::GpuMat & shift_2_1_, const cv::cuda::GpuMat & shift_2_2_, const cv::cuda::GpuMat & gray_2_,
            const cv::cuda::GpuMat& shift_3_0_, const cv::cuda::GpuMat & shift_3_1_, const cv::cuda::GpuMat & shift_3_2_, const cv::cuda::GpuMat & gray_3_,
            const int rows, const int cols,
            cv::cuda::GpuMat & wrapImg1_1_, cv::cuda::GpuMat & wrapImg1_2_, cv::cuda::GpuMat & wrapImg1_3_, cv::cuda::GpuMat & conditionImg_1_,
            cv::cuda::GpuMat & wrapImg2_1_, cv::cuda::GpuMat & wrapImg2_2_, cv::cuda::GpuMat & wrapImg2_3_, cv::cuda::GpuMat & conditionImg_2_,
            cv::cuda::GpuMat & wrapImg3_1_, cv::cuda::GpuMat & wrapImg3_2_, cv::cuda::GpuMat & wrapImg3_3_, cv::cuda::GpuMat & conditionImg_3_,
            cv::cuda::GpuMat & wrapImg4_1_, cv::cuda::GpuMat & wrapImg4_2_, cv::cuda::GpuMat & wrapImg4_3_, cv::cuda::GpuMat & conditionImg_4_,
            cv::cuda::GpuMat & floor_K, const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            atan3M_DevideSpace << <grid, block, 0, stream >> > (shift_0_0_, shift_0_1_, shift_0_2_, gray_0_,
                shift_1_0_, shift_1_1_, shift_1_2_, gray_1_,
                shift_2_0_, shift_2_1_, shift_2_2_, gray_2_,
                shift_3_0_, shift_3_1_, shift_3_2_, gray_3_,
                rows, cols,
                wrapImg1_1_, wrapImg1_2_, wrapImg1_3_, conditionImg_1_,
                wrapImg2_1_, wrapImg2_2_, wrapImg2_3_, conditionImg_2_,
                wrapImg3_1_, wrapImg3_2_, wrapImg3_3_, conditionImg_3_,
                wrapImg4_1_, wrapImg4_2_, wrapImg4_3_, conditionImg_4_, floor_K);
            /*
            cudaDeviceSynchronize();
            cv::Mat test[5];
            wrapImg1_1_.download(test[0]);
            wrapImg1_2_.download(test[1]);
            wrapImg1_3_.download(test[2]);
            floor_K.download(test[3]);
            conditionImg_1_.download(test[4]);
            */
        }

        void solvePhase_DevidedSpace(const cv::cuda::GpuMat & absolutImgWhite, const int rows, const int cols,
            const cv::cuda::GpuMat & wrapImg_1_0_, const cv::cuda::GpuMat & wrapImg_1_1_, const cv::cuda::GpuMat & wrapImg_1_2_, const cv::cuda::GpuMat & conditionImg_1_, cv::cuda::GpuMat & unwrapImg_1_,
            const cv::cuda::GpuMat & wrapImg_2_0_, const cv::cuda::GpuMat & wrapImg_2_1_, const cv::cuda::GpuMat & wrapImg_2_2_, const cv::cuda::GpuMat & conditionImg_2_, cv::cuda::GpuMat & unwrapImg_2_,
            const cv::cuda::GpuMat & wrapImg_3_0_, const cv::cuda::GpuMat & wrapImg_3_1_, const cv::cuda::GpuMat & wrapImg_3_2_, const cv::cuda::GpuMat & conditionImg_3_, cv::cuda::GpuMat & unwrapImg_3_,
            const cv::cuda::GpuMat & wrapImg_4_0_, const cv::cuda::GpuMat & wrapImg_4_1_, const cv::cuda::GpuMat & wrapImg_4_2_, const cv::cuda::GpuMat & conditionImg_4_, cv::cuda::GpuMat & unwrapImg_4_,
            cv::cuda::GpuMat & floor_K, const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            getUnwrapImg_DevidedSpace << <grid, block ,0, stream >> > (absolutImgWhite, rows, cols,
                wrapImg_1_0_, wrapImg_1_1_, wrapImg_1_2_, conditionImg_1_, unwrapImg_1_,
                wrapImg_2_0_, wrapImg_2_1_, wrapImg_2_2_, conditionImg_2_, unwrapImg_2_,
                wrapImg_3_0_, wrapImg_3_1_, wrapImg_3_2_, conditionImg_3_, unwrapImg_3_,
                wrapImg_4_0_, wrapImg_4_1_, wrapImg_4_2_, conditionImg_4_, unwrapImg_4_, floor_K);
            /*
            cudaDeviceSynchronize();
            cv::Mat test[7];
            absolutImgWhite.download(test[0]);
            wrapImg_1_0_.download(test[1]);
            wrapImg_1_1_.download(test[2]);
            wrapImg_1_2_.download(test[3]);
            conditionImg_1_.download(test[4]);
            floor_K.download(test[5]);
            unwrapImg_1_.download(test[6]);
            */
        }


        void solvePhasePrepare_ShiftGray(const cv::cuda::GpuMat& shift_0_0_, const cv::cuda::GpuMat& shift_0_1_, const cv::cuda::GpuMat& shift_0_2_,
            const cv::cuda::GpuMat& shift_1_0_, const cv::cuda::GpuMat& shift_1_1_, const cv::cuda::GpuMat& shift_1_2_,
            const cv::cuda::GpuMat& gray_0_, const cv::cuda::GpuMat& gray_1_, const cv::cuda::GpuMat& gray_2_, const cv::cuda::GpuMat& gray_3_,
            const int rows, const int cols,
            cv::cuda::GpuMat& wrapImg1, cv::cuda::GpuMat& conditionImg_1_,
            cv::cuda::GpuMat& wrapImg2, cv::cuda::GpuMat& conditionImg_2_,
            cv::cuda::GpuMat& floor_K, const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            atan3M_ShiftGray << <grid, block,0, stream >> > (shift_0_0_, shift_0_1_, shift_0_2_,
                shift_1_0_, shift_1_1_, shift_1_2_,
                gray_0_, gray_1_, gray_2_, gray_3_,
                rows, cols,
                wrapImg1, conditionImg_1_,
                wrapImg2, conditionImg_2_,floor_K);
        }

        /** \AbsolutImgWhite��������ƽ����չ������ʹ�� **/
        void solvePhase_ShiftGray(const cv::cuda::GpuMat & absolutImgWhite,const int rows, const int cols,
            const cv::cuda::GpuMat & wrapImg_1_, const cv::cuda::GpuMat & conditionImg_1_, cv::cuda::GpuMat & unwrapImg_1_,
            const cv::cuda::GpuMat & wrapImg_2_, const cv::cuda::GpuMat & conditionImg_2_, cv::cuda::GpuMat & unwrapImg_2_,cv::cuda::GpuMat & floor_K,
            const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            getUnwrapImg_ShiftGray<< <grid, block ,0, stream >> > (absolutImgWhite, rows, cols,
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

        void solvePhasePrepare_FourStepSixGray(const cv::cuda::GpuMat& shift_0_, const cv::cuda::GpuMat& shift_1_, const cv::cuda::GpuMat& shift_2_, const cv::cuda::GpuMat& shift_3_,
                                               const int rows, const int cols, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& averageImg, cv::cuda::GpuMat& conditionImg, 
                                               const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            atan2M_FourStepSixGray<<<grid, block, 0, stream >>>(shift_0_,shift_1_,shift_2_,shift_3_,
                                                                rows,cols,wrapImg,averageImg,conditionImg);
        }

        /** \AbsolutImgWhite��������ƽ����չ������ʹ�� **/
        void solvePhase_FourStepSixGray(const cv::cuda::GpuMat& Gray_0_, const cv::cuda::GpuMat& Gray_1_, const cv::cuda::GpuMat& Gray_2_, const cv::cuda::GpuMat& Gray_3_, const cv::cuda::GpuMat& Gray_4_, const cv::cuda::GpuMat& Gray_5_,
                                        const int rows, const int cols, const cv::cuda::GpuMat& averageImg, const cv::cuda::GpuMat& conditionImg, const cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& unwrapImg,
                                        const dim3 block, cv::cuda::Stream& cvStream) {
            cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
            dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
            getUnwrapImg_FourStepSixGray<<<grid, block,0, stream >>>(Gray_0_,Gray_1_,Gray_2_,Gray_3_,Gray_4_,Gray_5_,
                                         rows,cols,averageImg,conditionImg,wrapImg,unwrapImg);
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
    }
}
