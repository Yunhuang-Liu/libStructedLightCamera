#include <phaseSolver/fourStepSixGrayCodeMaster_CPU.h>

namespace sl {
    namespace phaseSolver {
        FourStepSixGrayCodeMaster_CPU::FourStepSixGrayCodeMaster_CPU(
                const cv::Mat &refImg, const int threads_) : threads(threads_),
                                                             refAbsImg(refImg) {
        }

        FourStepSixGrayCodeMaster_CPU::~FourStepSixGrayCodeMaster_CPU() {
        }

        void FourStepSixGrayCodeMaster_CPU::getWrapPhaseImg() {
            wrapImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0.0));
            atan2M(imgs[3] - imgs[1], imgs[0] - imgs[2], wrapImg, threads);
        }

        void FourStepSixGrayCodeMaster_CPU::caculateAverageImgs() {
            const int rows_ = imgs[0].rows;
            const int cols_ = imgs[0].cols;
            std::vector<std::thread> tasks;
            tasks.resize(threads);
            int rows = rows_ / threads;
            for (int i = 0; i < threads - 1; i++) {
                tasks[i] = std::thread(
                        &FourStepSixGrayCodeMaster_CPU::dataInit_Thread_SIMD,
                        this,
                        rows_,
                        cols_,
                        cv::Point2i(rows * i, rows * (i + 1)));
            }

            tasks[threads - 1] = std::thread(
                    &FourStepSixGrayCodeMaster_CPU::dataInit_Thread_SIMD,
                    this,
                    rows_,
                    cols_,
                    cv::Point2i(rows * (threads - 1), rows_));

            for (int i = 0; i < threads; i++) {
                if (tasks[i].joinable()) {
                    tasks[i].join();
                }
            }
        }

        void FourStepSixGrayCodeMaster_CPU::dataInit_Thread_SIMD(
                const int rows, const int cols, const cv::Point2i region) {
            __m256 img_0_data;
            __m256 img_1_data;
            __m256 img_2_data;
            __m256 img_3_data;
            __m256 value_4_data = _mm256_set1_ps(4);
            __m256 average_data;
            __m256 i3_i1_data;
            __m256 i0_i2_data;
            __m256 value_2_data = _mm256_set1_ps(2);
            __m256 condition_data;
            for (int i = region.x; i < region.y; i++) {
                const float *ptr_img_0_ = imgs[0].ptr<float>(i);
                const float *ptr_img_1_ = imgs[1].ptr<float>(i);
                const float *ptr_img_2_ = imgs[2].ptr<float>(i);
                const float *ptr_img_3_ = imgs[3].ptr<float>(i);
                float *ptr_averageImg = averageImg.ptr<float>(i);
                float *ptr_conditionImg = conditionImg.ptr<float>(i);
                for (int j = 0; j < cols; j += 8) {
                    img_0_data = _mm256_load_ps(&ptr_img_0_[j]);
                    img_1_data = _mm256_load_ps(&ptr_img_1_[j]);
                    img_2_data = _mm256_load_ps(&ptr_img_2_[j]);
                    img_3_data = _mm256_load_ps(&ptr_img_3_[j]);
                    __m256 add_0_1_ = _mm256_add_ps(img_0_data, img_1_data);
                    __m256 add_2_3_ = _mm256_add_ps(img_2_data, img_3_data);
                    __m256 add_avg = _mm256_add_ps(add_0_1_, add_2_3_);
                    average_data = _mm256_div_ps(add_avg, value_4_data);
                    i3_i1_data = _mm256_sub_ps(img_3_data, img_1_data);
                    i0_i2_data = _mm256_sub_ps(img_0_data, img_2_data);
                    __m256 pow_3_1_ = _mm256_mul_ps(i3_i1_data, i3_i1_data);
                    __m256 pow_0_2_ = _mm256_mul_ps(i0_i2_data, i0_i2_data);
                    __m256 add_cond = _mm256_add_ps(pow_3_1_, pow_0_2_);
                    condition_data = _mm256_div_ps(_mm256_sqrt_ps(add_cond), value_2_data);
                    _mm256_store_ps(&ptr_averageImg[j], average_data);
                    _mm256_store_ps(&ptr_conditionImg[j], condition_data);
                }
            }
        }

        void FourStepSixGrayCodeMaster_CPU::mutiplyThreadUnwrap(
                const int rows, const int cols,
                const cv::Point2i region, cv::Mat &absolutePhaseImg) {
            __m256 leftMove_5_ = _mm256_set1_ps(32);
            __m256 leftMove_4_ = _mm256_set1_ps(16);
            __m256 leftMove_3_ = _mm256_set1_ps(8);
            __m256 leftMove_2_ = _mm256_set1_ps(4);
            __m256 leftMove_1_ = _mm256_set1_ps(2);
            __m256 add_1_ = _mm256_set1_ps(1);
            __m256 div_2_ = _mm256_set1_ps(2);
            __m256 compare_Condition_10 = _mm256_set1_ps(5.0);
            __m256 K1;
            __m256 K2;
            __m256 _Counter_PI_Div_2_ = _mm256_set1_ps(-CV_PI / 2);
            __m256 _PI_Div_2_ = _mm256_set1_ps(CV_PI / 2);
            __m256 _2PI_ = _mm256_set1_ps(CV_2PI);
            __m256 _32PI_ = _mm256_set1_ps(CV_2PI * 16);
            __m256 zero = _mm256_set1_ps(0);
            __m256 one = _mm256_set1_ps(1);
            __m256 img_0_Data;
            __m256 img_1_Data;
            __m256 img_2_Data;
            __m256 img_3_Data;
            __m256 img_4_Data;
            __m256 img_5_Data;
            __m256 img_refAbs_Data;
            __m256 averageData;
            __m256 wrapImgData;
            __m256 conditionData;
            __m256 compareCondition;
            __m256 compareImg_0_;
            __m256 compareImg_1_;
            __m256 compareImg_2_;
            __m256 compareImg_3_;
            __m256 compareImg_4_;
            __m256 compareImg_5_;
            __m256 condition_CompareData;
            __m256 Img_0_CompareData;
            __m256 Img_1_CompareData;
            __m256 Img_2_CompareData;
            __m256 Img_3_CompareData;
            __m256 Img_4_CompareData;
            __m256 Img_5_CompareData;
            __m256 bit5;
            __m256 bit4;
            __m256 bit3;
            __m256 bit2;
            __m256 bit1;
            __m256 bit0;
            __m256 lessEqualThan;
            __m256 greaterEqualThan;
            __m256 less_data_greaterThan;
            __m256 data_1_;
            __m256 data_2_;
            __m256 data;
            for (int i = region.x; i < region.y; i++) {
                const float *ptr0 = imgs[4].ptr<float>(i);
                const float *ptr1 = imgs[5].ptr<float>(i);
                const float *ptr2 = imgs[6].ptr<float>(i);
                const float *ptr3 = imgs[7].ptr<float>(i);
                const float *ptr4 = imgs[8].ptr<float>(i);
                const float *ptr5 = imgs[9].ptr<float>(i);
                const float *ptr_Average = averageImg.ptr<float>(i);
                const float *ptr_WrapImg = wrapImg.ptr<float>(i);
                const float *ptr_Condition = conditionImg.ptr<float>(i);
                float *ptr_absoluteImg = absolutePhaseImg.ptr<float>(i);
                const float *ptr_refAbsImg = nullptr;
                if (!refAbsImg.empty()) {
                    ptr_refAbsImg = refAbsImg.ptr<float>(i);
                }
                for (int j = 0; j < cols; j += 8) {
                    img_0_Data = _mm256_load_ps(&ptr0[j]);
                    img_1_Data = _mm256_load_ps(&ptr1[j]);
                    img_2_Data = _mm256_load_ps(&ptr2[j]);
                    img_3_Data = _mm256_load_ps(&ptr3[j]);
                    img_4_Data = _mm256_load_ps(&ptr4[j]);
                    img_5_Data = _mm256_load_ps(&ptr5[j]);
                    averageData = _mm256_load_ps(&ptr_Average[j]);
                    wrapImgData = _mm256_load_ps(&ptr_WrapImg[j]);
                    conditionData = _mm256_load_ps(&ptr_Condition[j]);
                    compareCondition = _mm256_cmp_ps(conditionData, compare_Condition_10, _CMP_GT_OS);
                    compareImg_0_ = _mm256_cmp_ps(img_0_Data, averageData, _CMP_GE_OS);
                    compareImg_1_ = _mm256_cmp_ps(img_1_Data, averageData, _CMP_GE_OS);
                    compareImg_2_ = _mm256_cmp_ps(img_2_Data, averageData, _CMP_GE_OS);
                    compareImg_3_ = _mm256_cmp_ps(img_3_Data, averageData, _CMP_GE_OS);
                    compareImg_4_ = _mm256_cmp_ps(img_4_Data, averageData, _CMP_GE_OS);
                    compareImg_5_ = _mm256_cmp_ps(img_5_Data, averageData, _CMP_GE_OS);
                    Img_0_CompareData = _mm256_and_ps(compareImg_0_, one);
                    Img_1_CompareData = _mm256_and_ps(compareImg_1_, one);
                    Img_2_CompareData = _mm256_and_ps(compareImg_2_, one);
                    Img_3_CompareData = _mm256_and_ps(compareImg_3_, one);
                    Img_4_CompareData = _mm256_and_ps(compareImg_4_, one);
                    Img_5_CompareData = _mm256_and_ps(compareImg_5_, one);
                    bit5 = _mm256_xor_ps(Img_0_CompareData, zero);
                    bit4 = _mm256_xor_ps(Img_1_CompareData, bit5);
                    bit3 = _mm256_xor_ps(Img_2_CompareData, bit4);
                    bit2 = _mm256_xor_ps(Img_3_CompareData, bit3);
                    bit1 = _mm256_xor_ps(Img_4_CompareData, bit2);
                    bit0 = _mm256_xor_ps(Img_5_CompareData, bit1);
                    K2 = _mm256_floor_ps(_mm256_div_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(bit5, leftMove_5_), _mm256_mul_ps(bit4, leftMove_4_)), _mm256_mul_ps(bit3, leftMove_3_)),
                                                                                                               _mm256_mul_ps(bit2, leftMove_2_)),
                                                                                                 _mm256_mul_ps(bit1, leftMove_1_)),
                                                                                   bit0),
                                                                     add_1_),
                                                       div_2_));
                    bit4 = _mm256_xor_ps(Img_0_CompareData, zero);
                    bit3 = _mm256_xor_ps(Img_1_CompareData, bit4);
                    bit2 = _mm256_xor_ps(Img_2_CompareData, bit3);
                    bit1 = _mm256_xor_ps(Img_3_CompareData, bit2);
                    bit0 = _mm256_xor_ps(Img_4_CompareData, bit1);
                    K1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(bit4, leftMove_4_), _mm256_mul_ps(bit3, leftMove_3_)), _mm256_mul_ps(bit2, leftMove_2_)),
                                                     _mm256_mul_ps(bit1, leftMove_1_)),
                                       bit0);
                    lessEqualThan = _mm256_and_ps(_mm256_cmp_ps(wrapImgData, _Counter_PI_Div_2_, _CMP_LE_OS), one);
                    greaterEqualThan = _mm256_and_ps(_mm256_cmp_ps(wrapImgData, _PI_Div_2_, _CMP_GE_OS), one);
                    less_data_greaterThan = _mm256_xor_ps(_mm256_or_ps(lessEqualThan, greaterEqualThan), one);
                    data_1_ = _mm256_mul_ps(lessEqualThan, _mm256_fmadd_ps(_2PI_, K2, wrapImgData));
                    data_2_ = _mm256_mul_ps(greaterEqualThan, _mm256_fmadd_ps(_2PI_, _mm256_sub_ps(K2, one), wrapImgData));
                    data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(less_data_greaterThan, _mm256_fmadd_ps(_2PI_, K1, wrapImgData)), data_1_), data_2_);
                    if (ptr_refAbsImg != nullptr) {
                        img_refAbs_Data = _mm256_load_ps(&ptr_refAbsImg[j]);
                        data = _mm256_add_ps(_mm256_mul_ps(_mm256_floor_ps(_mm256_div_ps(_mm256_sub_ps(img_refAbs_Data, data), _32PI_)), _32PI_), data);
                    }
                    _mm256_store_ps(&ptr_absoluteImg[j], _mm256_mul_ps(data, _mm256_and_ps(compareCondition, one)));
                }
            }
        }

        void FourStepSixGrayCodeMaster_CPU::getUnwrapPhaseImg(cv::Mat &absolutePhaseImg) {
            std::vector<std::thread> convertFloatThreads(imgs.size());
            for (int i = 0; i < convertFloatThreads.size(); i++) {
                convertFloatThreads[i] = std::thread([&, i] {
                    imgs[i].convertTo(imgs[i], CV_32FC1);
                });
            }
            for (auto &thread: convertFloatThreads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            absolutePhaseImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            conditionImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            averageImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            getWrapPhaseImg();
            caculateAverageImgs();
            const int rows_ = absolutePhaseImg.rows;
            const int cols_ = absolutePhaseImg.cols;
            std::vector<std::thread> tasks;
            tasks.resize(threads);
            int rows = rows_ / threads;
            for (int i = 0; i < threads - 1; i++) {
                tasks[i] = std::thread(
                        &FourStepSixGrayCodeMaster_CPU::mutiplyThreadUnwrap,
                        this,
                        rows_,
                        cols_,
                        cv::Point2i(rows * i, rows * (i + 1)),
                        std::ref(absolutePhaseImg));
            }

            tasks[threads - 1] = std::thread(
                    &FourStepSixGrayCodeMaster_CPU::mutiplyThreadUnwrap,
                    this,
                    rows_,
                    cols_,
                    cv::Point2i(rows * (threads - 1), rows_),
                    std::ref(absolutePhaseImg));

            for (int i = 0; i < threads; i++) {
                tasks[i].join();
            }
        }

        void FourStepSixGrayCodeMaster_CPU::changeSourceImg(
                std::vector<cv::Mat> &imgs_) {
            imgs = imgs_;
        }

        void FourStepSixGrayCodeMaster_CPU::atan2M(
                const cv::Mat &lhs, const cv::Mat &rhs, cv::Mat &wrapImg, const int threads) {
            std::vector<std::thread> tasks;
            tasks.resize(threads);
            int rows = lhs.rows / threads;
            for (int i = 0; i < threads - 1; i++) {
                tasks[i] = std::thread(
                        &phaseSolver::FourStepSixGrayCodeMaster_CPU::SIMD_WrapImg,
                        this,
                        std::ref(lhs),
                        std::ref(rhs),
                        cv::Point2i(rows * i, rows * (i + 1)),
                        std::ref(wrapImg));
            }
            tasks[threads - 1] = std::thread(
                    &phaseSolver::FourStepSixGrayCodeMaster_CPU::SIMD_WrapImg,
                    this,
                    std::ref(lhs),
                    std::ref(rhs),
                    cv::Point2i(rows * (threads - 1), lhs.rows),
                    std::ref(wrapImg));
            for (int i = 0; i < tasks.size(); i++) {
                if (tasks[i].joinable()) {
                    tasks[i].join();
                }
            }
        }

        void FourStepSixGrayCodeMaster_CPU::SIMD_WrapImg(
                const cv::Mat &lhs,
                const cv::Mat &rhs,
                const cv::Point2i &region,
                cv::Mat &wrapImg) {
            __m256 lhs_Up;
            __m256 rhs_Down;
            const int cols = lhs.cols;
            for (size_t i = region.x; i < region.y; i++) {
                const float *ptr_lhs = lhs.ptr<float>(i);
                const float *ptr_rhs = rhs.ptr<float>(i);
                float *ptr_wrapImg = wrapImg.ptr<float>(i);
                for (size_t j = 0; j < cols; j += 8) {
                    lhs_Up = _mm256_load_ps(&ptr_lhs[j]);
                    rhs_Down = _mm256_load_ps(&ptr_rhs[j]);
                    for (int d = 0; d < 8; d++) {
                        ptr_wrapImg[j + d] = atan2f(lhs_Up.m256_f32[d], rhs_Down.m256_f32[d]);
                    }
                }
            }
        }

        FourStepSixGrayCodeMaster_CPU::FourStepSixGrayCodeMaster_CPU(
                std::vector<cv::Mat> &imgs_, const cv::Mat &refImg_, const int threads_) : imgs(imgs_), refAbsImg(refImg_), threads(threads_) {
        }

        void FourStepSixGrayCodeMaster_CPU::getWrapPhaseImg(
                cv::Mat &wrapImg_, cv::Mat &conditionImg_) {
            std::vector<std::thread> convertFloatThreads(imgs.size());
            for (int i = 0; i < convertFloatThreads.size(); i++) {
                convertFloatThreads[i] = std::thread([&, i] {
                    imgs[i].convertTo(imgs[i], CV_32FC1);
                });
            }
            for (auto &thread: convertFloatThreads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            getWrapPhaseImg();
            conditionImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            averageImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            caculateAverageImgs();
            wrapImg_ = wrapImg;
            conditionImg_ = conditionImg;
        }

        void FourStepSixGrayCodeMaster_CPU::getTextureImg(cv::Mat &textureImg) {
            textureImg = averageImg;
        }
    }// namespace phaseSolver
}// namespace sl
