#include <phaseSolver/nStepNGrayCodeMaster_CPU.h>

namespace sl {
    namespace phaseSolver {
        NStepNGrayCodeMaster_CPU::NStepNGrayCodeMaster_CPU(
                const int shiftStep_, const int threads_) : shiftStep(shiftStep_),
                                                            threads(threads_) {
        }

        NStepNGrayCodeMaster_CPU::~NStepNGrayCodeMaster_CPU() {
        }

        void NStepNGrayCodeMaster_CPU::getWrapPhaseImg() {
            atan2M(imgs, wrapImg, shiftStep, threads);
        }

        void NStepNGrayCodeMaster_CPU::caculateAverageImgs() {
            const int rows_ = imgs[0].rows;
            const int cols_ = imgs[0].cols;
            std::vector<std::thread> tasks;
            tasks.resize(threads);
            int rows = rows_ / threads;
            for (int i = 0; i < threads - 1; i++) {
                tasks[i] = std::thread(
                        &NStepNGrayCodeMaster_CPU::dataInit_Thread_SIMD,
                        this,
                        std::ref(imgs),
                        shiftStep,
                        std::ref(wrapImg),
                        cv::Point2i(rows * i, rows * (i + 1)),
                        std::ref(conditionImg),
                        std::ref(averageImg));
            }

            tasks[threads - 1] = std::thread(
                    &NStepNGrayCodeMaster_CPU::dataInit_Thread_SIMD,
                    this,
                    std::ref(imgs),
                    shiftStep,
                    std::ref(wrapImg),
                    cv::Point2i(rows * (threads - 1), rows_),
                    std::ref(conditionImg),
                    std::ref(averageImg));

            for (int i = 0; i < threads; i++) {
                if (tasks[i].joinable()) {
                    tasks[i].join();
                }
            }
        }

        void NStepNGrayCodeMaster_CPU::dataInit_Thread_SIMD(
                const std::vector<cv::Mat> &imgs,
                const int shiftStep,
                const cv::Mat &wrapImg,
                const cv::Point2i region,
                cv::Mat &conditionImg,
                cv::Mat &textureImg) {
            __m256 shiftStepData = _mm256_set1_ps(shiftStep);
            __m256 sumShiftImg = _mm256_set1_ps(0);
            __m256 perShiftDistance = _mm256_set1_ps(CV_2PI / shiftStep);
            __m256 bCosData = _mm256_set1_ps(0);
            __m256 cosData = _mm256_set1_ps(0);
            __m256 conditon = _mm256_set1_ps(0);
            __m256 zero = _mm256_set1_ps(0);
            __m256 one = _mm256_set1_ps(1);
            __m256 maxVal = _mm256_set1_ps(255);
            std::vector<__m256> dataImgs(shiftStep);
            const int rows = conditionImg.rows;
            const int cols = conditionImg.cols;
            for (int i = region.x; i < region.y; i++) {
                std::vector<const float *> ptrImgs(shiftStep);
                for (int step = 0; step < shiftStep; ++step)
                    ptrImgs[step] = imgs[step].ptr<float>(i);
                const float *ptrWrapImg = wrapImg.ptr<float>(i);
                float *ptr_averageImg = textureImg.ptr<float>(i);
                float *ptr_conditionImg = conditionImg.ptr<float>(i);
                for (int j = 0; j < cols; j += 8) {
                    sumShiftImg = _mm256_set1_ps(0);
                    for (int step = 0; step < shiftStep; ++step) {
                        dataImgs[step] = _mm256_load_ps(&ptrImgs[step][j]);
                        sumShiftImg = _mm256_add_ps(sumShiftImg, dataImgs[step]);
                    }
                    sumShiftImg = _mm256_div_ps(sumShiftImg, shiftStepData);
                    _mm256_store_ps(&ptr_averageImg[j], sumShiftImg);

                    bCosData = _mm256_sub_ps(dataImgs[0], sumShiftImg);
                    cosData = _mm256_cos_ps(_mm256_load_ps(&ptrWrapImg[j]));
                    conditon = _mm256_div_ps(bCosData, cosData);
                    __m256 greaterZero = _mm256_and_ps(_mm256_cmp_ps(conditon, zero, _CMP_GE_OS), one);
                    __m256 lessMaxVal = _mm256_and_ps(_mm256_cmp_ps(conditon, maxVal, _CMP_LE_OS), one);
                    __m256 filterFlag = _mm256_and_ps(greaterZero, lessMaxVal);
                    _mm256_store_ps(&ptr_conditionImg[j], _mm256_mul_ps(conditon, filterFlag));
                }
            }
        }

        void NStepNGrayCodeMaster_CPU::mutiplyThreadUnwrap(
                const std::vector<cv::Mat> &imgs,
                const int shiftStep,
                const cv::Mat &conditionImg,
                const cv::Mat &wrapImg,
                const cv::Point2i region,
                cv::Mat &absolutePhaseImg) {
            const int rows = absolutePhaseImg.rows;
            const int cols = absolutePhaseImg.cols;
            __m256 add_1_ = _mm256_set1_ps(1);
            __m256 div_2_ = _mm256_set1_ps(2);
            __m256 compare_Condition_10 = _mm256_set1_ps(10.0);
            __m256 _Counter_PI_Div_2_ = _mm256_set1_ps(-CV_PI / 2);
            __m256 _PI_Div_2_ = _mm256_set1_ps(CV_PI / 2);
            __m256 _2PI_ = _mm256_set1_ps(CV_2PI);
            __m256 zero = _mm256_set1_ps(0);
            __m256 one = _mm256_set1_ps(1);
            std::vector<__m256> leftMoveTime(imgs.size() - shiftStep);
            for (int gray = 0; gray < imgs.size() - shiftStep; ++gray)
                leftMoveTime[gray] = _mm256_set1_ps(pow(2, gray));
            std::vector<__m256> compareImgData(imgs.size() - shiftStep);
            std::vector<__m256> compareData(imgs.size() - shiftStep);
            std::vector<__m256> bitData(imgs.size() - shiftStep);
            std::vector<__m256> imgsData(imgs.size() - shiftStep);
            for (int i = region.x; i < region.y; i++) {
                std::vector<const float *> ptrImgs(imgs.size() - shiftStep);
                for (int gray = shiftStep; gray < imgs.size(); ++gray)
                    ptrImgs[gray - shiftStep] = imgs[gray].ptr<float>(i);
                const float *ptr_WrapImg = wrapImg.ptr<float>(i);
                const float *ptr_Average = averageImg.ptr<float>(i);
                const float *ptr_Condition = conditionImg.ptr<float>(i);
                float *ptr_absoluteImg = absolutePhaseImg.ptr<float>(i);
                for (int j = 0; j < cols; j += 8) {
                    for (int gray = 0; gray < imgs.size() - shiftStep; ++gray) {
                        imgsData[gray] = _mm256_load_ps(&ptrImgs[gray][j]);
                    }
                    __m256 averageData = _mm256_load_ps(&ptr_Average[j]);
                    __m256 wrapImgData = _mm256_load_ps(&ptr_WrapImg[j]);
                    __m256 conditionData = _mm256_load_ps(&ptr_Condition[j]);
                    __m256 compareCondition = _mm256_cmp_ps(conditionData, compare_Condition_10, _CMP_GT_OS);
                    for (int gray = 0; gray < compareImgData.size(); ++gray) {
                        compareImgData[gray] = _mm256_and_ps(_mm256_cmp_ps(imgsData[gray], averageData, _CMP_GE_OS), one);
                    }
                    __m256 sumDataK2 = _mm256_set1_ps(0);
                    __m256 sumDataK1 = _mm256_set1_ps(0);
                    for (int gray = compareImgData.size() - 1; gray >= 0; --gray) {
                        if (gray == compareImgData.size() - 1)
                            bitData[gray] = _mm256_xor_ps(compareImgData[compareImgData.size() - 1 - gray], zero);
                        else
                            bitData[gray] = _mm256_xor_ps(compareImgData[compareImgData.size() - 1 - gray], bitData[gray + 1]);
                        sumDataK2 = _mm256_add_ps(sumDataK2, _mm256_mul_ps(bitData[gray], leftMoveTime[gray]));
                        if (gray - 1 >= 0)
                            sumDataK1 = _mm256_add_ps(sumDataK1, _mm256_mul_ps(bitData[gray], leftMoveTime[gray - 1]));
                    }
                    __m256 K2 = _mm256_floor_ps(_mm256_div_ps(_mm256_add_ps(sumDataK2, add_1_), div_2_));
                    __m256 K1 = sumDataK1;
                    __m256 lessEqualThan = _mm256_and_ps(_mm256_cmp_ps(wrapImgData, _Counter_PI_Div_2_, _CMP_LE_OS), one);
                    __m256 greaterEqualThan = _mm256_and_ps(_mm256_cmp_ps(wrapImgData, _PI_Div_2_, _CMP_GE_OS), one);
                    __m256 less_data_greaterThan = _mm256_xor_ps(_mm256_or_ps(lessEqualThan, greaterEqualThan), one);
                    __m256 data_1_ = _mm256_mul_ps(lessEqualThan, _mm256_fmadd_ps(_2PI_, K2, wrapImgData));
                    __m256 data_2_ = _mm256_mul_ps(greaterEqualThan, _mm256_fmadd_ps(_2PI_, _mm256_sub_ps(K2, one), wrapImgData));
                    __m256 data = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(less_data_greaterThan, _mm256_fmadd_ps(_2PI_, K1, wrapImgData)), data_1_), data_2_);
                    _mm256_store_ps(&ptr_absoluteImg[j], _mm256_mul_ps(data, _mm256_and_ps(compareCondition, one)));
                }
            }
        }

        void NStepNGrayCodeMaster_CPU::getUnwrapPhaseImg(cv::Mat &absolutePhaseImg) {
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
            wrapImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            getWrapPhaseImg();
            caculateAverageImgs();
            const int rows_ = absolutePhaseImg.rows;
            const int cols_ = absolutePhaseImg.cols;
            std::vector<std::thread> tasks;
            tasks.resize(threads);
            int rows = rows_ / threads;
            for (int i = 0; i < threads - 1; i++) {
                tasks[i] = std::thread(
                        &NStepNGrayCodeMaster_CPU::mutiplyThreadUnwrap,
                        this,
                        std::ref(imgs),
                        shiftStep,
                        std::ref(conditionImg),
                        std::ref(wrapImg),
                        cv::Point2i(rows * i, rows * (i + 1)),
                        std::ref(absolutePhaseImg));
            }

            tasks[threads - 1] = std::thread(
                    &NStepNGrayCodeMaster_CPU::mutiplyThreadUnwrap,
                    this,
                    std::ref(imgs),
                    shiftStep,
                    std::ref(conditionImg),
                    std::ref(wrapImg),
                    cv::Point2i(rows * (threads - 1), rows_),
                    std::ref(absolutePhaseImg));

            for (int i = 0; i < threads; i++) {
                tasks[i].join();
            }
        }

        void NStepNGrayCodeMaster_CPU::changeSourceImg(
                std::vector<cv::Mat> &imgs_) {
            imgs = imgs_;
        }

        void NStepNGrayCodeMaster_CPU::atan2M(
                const std::vector<cv::Mat> &imgs, cv::Mat &wrapImg, const int shiftStep, const int threads) {
            std::vector<std::thread> tasks;
            tasks.resize(threads);
            int rows = wrapImg.rows / threads;
            for (int i = 0; i < threads - 1; i++) {
                tasks[i] = std::thread(
                        &phaseSolver::NStepNGrayCodeMaster_CPU::SIMD_WrapImg,
                        this,
                        std::ref(imgs),
                        shiftStep,
                        cv::Point2i(rows * i, rows * (i + 1)),
                        std::ref(wrapImg));
            }
            tasks[threads - 1] = std::thread(
                    &phaseSolver::NStepNGrayCodeMaster_CPU::SIMD_WrapImg,
                    this,
                    std::ref(imgs),
                    shiftStep,
                    cv::Point2i(rows * (threads - 1), wrapImg.rows),
                    std::ref(wrapImg));
            for (int i = 0; i < tasks.size(); i++) {
                if (tasks[i].joinable()) {
                    tasks[i].join();
                }
            }
        }

        void NStepNGrayCodeMaster_CPU::SIMD_WrapImg(
                const std::vector<cv::Mat> &imgs,
                const int shiftStep,
                const cv::Point2i &region,
                cv::Mat &wrapImg) {
            __m256 dataCounter1 = _mm256_set1_ps(-1.f);
            __m256 shiftDistance = _mm256_set1_ps(CV_2PI / shiftStep);
            const int cols = wrapImg.cols;
            std::vector<__m256> datas(shiftStep);
            __m256 sinPartial = _mm256_set1_ps(0);
            __m256 cosPartial = _mm256_set1_ps(0);
            __m256 time = _mm256_set1_ps(0);
            __m256 shiftNow = _mm256_set1_ps(0);
            __m256 result = _mm256_set1_ps(0);
            for (size_t i = region.x; i < region.y; i++) {
                std::vector<const float *> ptrImgs(shiftStep);
                for (int step = 0; step < shiftStep; ++step)
                    ptrImgs[step] = imgs[step].ptr<float>(i);
                float *ptr_wrapImg = wrapImg.ptr<float>(i);
                for (size_t j = 0; j < cols; j += 8) {
                    for (int step = 0; step < shiftStep; ++step)
                        datas[step] = _mm256_load_ps(&ptrImgs[step][j]);
                    sinPartial = _mm256_set1_ps(0);
                    cosPartial = _mm256_set1_ps(0);
                    for (int step = 0; step < shiftStep; ++step) {
                        time = _mm256_set1_ps(step);
                        shiftNow = _mm256_mul_ps(shiftDistance, time);
                        sinPartial = _mm256_add_ps(sinPartial, _mm256_mul_ps(datas[step], _mm256_sin_ps(shiftNow)));
                        cosPartial = _mm256_add_ps(cosPartial, _mm256_mul_ps(datas[step], _mm256_cos_ps(shiftNow)));
                    }
                    result = _mm256_mul_ps(dataCounter1, _mm256_atan2_ps(sinPartial, cosPartial));
                    _mm256_store_ps(&ptr_wrapImg[j], result);
                    /*
        for (int d = 0; d < 8; d++) {
          ptr_wrapImg[j + d] = atan2f(lhs_Up.m256_f32[d], rhs_Down.m256_f32[d]);
        }
        */
                }
            }
        }

        NStepNGrayCodeMaster_CPU::NStepNGrayCodeMaster_CPU(
                std::vector<cv::Mat> &imgs_, const int shiftStep_, const int threads_) : imgs(imgs_), shiftStep(shiftStep_), threads(threads_) {
        }

        void NStepNGrayCodeMaster_CPU::getWrapPhaseImg(
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
            conditionImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            averageImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            wrapImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0));
            getWrapPhaseImg();
            caculateAverageImgs();
            wrapImg_ = wrapImg;
            conditionImg_ = conditionImg;
        }

        void NStepNGrayCodeMaster_CPU::getTextureImg(cv::Mat &textureImg) {
            textureImg = averageImg;
        }
    }// namespace phaseSolver
}// namespace sl