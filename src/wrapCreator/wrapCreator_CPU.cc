#include <wrapCreator/wrapCreator_CPU.h>

namespace sl {
    namespace wrapCreator {
        WrapCreator_CPU::WrapCreator_CPU() {
        }

        WrapCreator_CPU::~WrapCreator_CPU() {
        }

        void WrapCreator_CPU::getWrapImg(
                const std::vector<cv::Mat> &imgs, cv::Mat &wrapImg,
                cv::Mat &conditionImg, const bool isCounter, const WrapParameter parameter) {
            if (imgs[0].empty()) {
                std::cout << "img is invalid" << std::endl;
                return;
            }

            const int numImg = imgs.size();
            const int rows = imgs[0].rows;
            const int cols = imgs[0].cols;
            const int threadsUsed = parameter.threads;

            std::vector<cv::Mat> copyImg(numImg);
            std::vector<std::thread> threadsConvertImg(numImg);
            for (int i = 0; i < numImg; i++) {
                threadsConvertImg[i] = std::thread([&, i] {
                    if (CV_8UC3 == imgs[i].type()) {
                        cv::cvtColor(imgs[i], copyImg[i], cv::COLOR_BGR2GRAY);
                        copyImg[i].convertTo(copyImg[i], CV_32FC1);
                    } else {
                        imgs[i].convertTo(copyImg[i], CV_32FC1);
                    }
                });
            }

            wrapImg.create(rows, cols, CV_32FC1);
            conditionImg.create(rows, cols, CV_32FC1);

            for (auto &thread: threadsConvertImg) {
                if (thread.joinable())
                    thread.join();
            }

            std::vector<std::thread> threads(threadsUsed);
            const int blockRows = wrapImg.rows / threadsUsed;
            for (int i = 0; i < threadsUsed; i++) {
                if (imgs.size() == 3)
                    threads[i] = std::thread(&wrapCreator::WrapCreator_CPU::thread_ThreeStepWrap, this, std::ref(copyImg), std::ref(wrapImg), std::ref(conditionImg), cv::Size(cols, blockRows * (i + 1)));
                else if (imgs.size() == 4)
                    threads[i] = std::thread(&wrapCreator::WrapCreator_CPU::thread_FourStepWrap, this, std::ref(copyImg), std::ref(wrapImg), std::ref(conditionImg), cv::Size(cols, blockRows * (i + 1)), isCounter);
                else
                    std::cout << "The " << numImg << " step is not support in current!" << std::endl;
            }

            for (auto &thread: threads) {
                if (thread.joinable())
                    thread.join();
            }
        }

        void WrapCreator_CPU::thread_ThreeStepWrap(
                const std::vector<cv::Mat> &imgs, cv::Mat &wrapImg,
                cv::Mat &conditionImg, const cv::Size region) {
            if (wrapImg.type() != CV_32FC1) {
                std::cout << "WRAP ERROR: This avx accelerate algorith is only support for CV_32FC1 image!" << std::endl;
                return;
            }

            __m256 dataTwo = _mm256_set1_ps(2.f);
            __m256 dataThree = _mm256_set1_ps(3.f);
            __m256 dataSqrtThree = _mm256_set1_ps(std::sqrt(3.f));

            for (int i = 0; i < region.height; i++) {
                const float *ptr_firstStep = imgs[0].ptr<float>(i);
                const float *ptr_secondStep = imgs[1].ptr<float>(i);
                const float *ptr_thirdStep = imgs[2].ptr<float>(i);

                float *ptr_wrapImg = wrapImg.ptr<float>(i);
                float *ptr_conditionImg = conditionImg.ptr<float>(i);

                for (int j = 0; j < region.width; j += 8) {
                    __m256 firsteStep = _mm256_load_ps(&ptr_firstStep[j]);
                    __m256 secondStep = _mm256_load_ps(&ptr_secondStep[j]);
                    __m256 thirdStep = _mm256_load_ps(&ptr_thirdStep[j]);

                    __m256 diffFT = _mm256_sub_ps(firsteStep, thirdStep);
                    __m256 diffSSFT = _mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(secondStep, dataTwo), firsteStep), thirdStep);

                    __m256 wrapVal = _mm256_atan2_ps(_mm256_mul_ps(diffFT, dataSqrtThree), diffSSFT);
                    _mm256_store_ps(&ptr_wrapImg[j], wrapVal);

                    __m256 powDiffFT = _mm256_mul_ps(diffFT, diffFT);
                    __m256 powDiffSSFT = _mm256_mul_ps(diffSSFT, diffSSFT);
                    __m256 conditionVal = _mm256_div_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(powDiffFT, dataThree), powDiffSSFT)), dataThree);
                    //Notice that we don't apply a filter for condition val which is less than threshod(eg. threshod =  5.f)
                    _mm256_store_ps(&ptr_conditionImg[j], conditionVal);
                }
            }
        }

        void WrapCreator_CPU::thread_FourStepWrap(
                const std::vector<cv::Mat> &imgs, cv::Mat &wrapImg,
                cv::Mat &conditionImg, const cv::Size region,
                const bool isCounter) {
            if (wrapImg.type() != CV_32FC1) {
                std::cout << "WRAP ERROR: This avx accelerate algorith is only support for CV_32FC1 image!" << std::endl;
                return;
            }

            __m256 dataTwo = _mm256_set1_ps(2.f);
            __m256 dataThree = _mm256_set1_ps(4.f);

            for (int i = 0; i < region.height; i++) {
                const float *ptr_firstStep = !isCounter ? imgs[0].ptr<float>(i) : imgs[2].ptr<float>(i);
                const float *ptr_secondStep = !isCounter ? imgs[1].ptr<float>(i) : imgs[3].ptr<float>(i);
                const float *ptr_thirdStep = !isCounter ? imgs[2].ptr<float>(i) : imgs[0].ptr<float>(i);
                const float *ptr_fourthStep = !isCounter ? imgs[3].ptr<float>(i) : imgs[1].ptr<float>(i);

                float *ptr_wrapImg = wrapImg.ptr<float>(i);
                float *ptr_conditionImg = conditionImg.ptr<float>(i);

                for (int j = 0; j < region.width; j += 8) {
                    __m256 firsteStep = _mm256_load_ps(&ptr_firstStep[j]);
                    __m256 secondStep = _mm256_load_ps(&ptr_secondStep[j]);
                    __m256 thirdStep = _mm256_load_ps(&ptr_thirdStep[j]);
                    __m256 fourthStep = _mm256_load_ps(&ptr_fourthStep[j]);

                    __m256 diffFS = _mm256_sub_ps(fourthStep, secondStep);
                    __m256 diffFT = _mm256_sub_ps(firsteStep, thirdStep);

                    __m256 wrapVal = _mm256_atan2_ps(diffFS, diffFT);
                    _mm256_store_ps(&ptr_wrapImg[j], wrapVal);

                    __m256 powDiffFS = _mm256_mul_ps(diffFS, diffFS);
                    __m256 powDiffFT = _mm256_mul_ps(diffFT, diffFT);
                    __m256 conditionVal = _mm256_div_ps(_mm256_sqrt_ps(_mm256_add_ps(powDiffFS, powDiffFT)), dataTwo);
                    //Notice that we don't apply a filter for condition val which is less than threshod(eg. threshod =  5.f)
                    _mm256_store_ps(&ptr_conditionImg[j], conditionVal);
                }
            }
        }
    }// namespace wrapCreator
}// namespace sl