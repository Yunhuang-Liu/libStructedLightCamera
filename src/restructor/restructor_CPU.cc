#include <Restructor/Restructor_CPU.h>

namespace sl {
    namespace restructor {
        Restructor_CPU::Restructor_CPU(
                const tool::Info &calibrationInfo_, const int minDisparity_, const int maxDisparity_,
                const float minDepth_, const float maxDepth_, const int threads_) : calibrationInfo(calibrationInfo_),
                                                                                    minDisparity(minDisparity_),
                                                                                    maxDisparity(maxDisparity_),
                                                                                    minDepth(minDepth_),
                                                                                    maxDepth(maxDepth_),
                                                                                    threads(threads_) {
            cv::Mat Q_CV;
            calibrationInfo_.Q.convertTo(Q_CV, CV_32FC1);
            /*
            //如果深度值被修改且视差值保持默认，则自动进行视差值的计算
            if (-500 == minDisparity && 500 == maxDisparity &&
                170 != minDepth && 220 != maxDepth) {
                float tx = -1.f / Q_CV.at<float>(3, 2);
                float crj = tx * Q_CV.at<float>(3, 3);
                float f = Q_CV.at<float>(2, 3);
                minDisparity = -tx * f / minDepth - crj;
                minDisparity = -tx * f / maxDepth - crj;
            }
            */
        }

        Restructor_CPU::~Restructor_CPU() {
        }

        void Restructor_CPU::restruction(const cv::Mat &leftAbsImg,
                                         const cv::Mat &rightAbsImg, cv::Mat &depthImgOut, const bool isMap, const bool isColor) {
            if (depthImgOut.empty())
                depthImgOut = cv::Mat(leftAbsImg.size(), CV_32FC1, cv::Scalar(0.f));
            else
                depthImgOut.setTo(0);
            getDepthColorMap(leftAbsImg, rightAbsImg, depthImgOut, isMap, isColor);
        }

        void Restructor_CPU::getDepthColorMap(
                const cv::Mat &leftAbsImg, const cv::Mat &rightAbsImg,
                cv::Mat &depthImgOut, const bool isMap, const bool isColor) {
            std::vector<std::thread> tasks;
            tasks.resize(threads);
            const int rows = leftAbsImg.rows / threads;
            for (int i = 0; i < threads - 1; i++) {
                tasks[i] = std::thread(
                        &restructor::Restructor_CPU::thread_DepthColorMap,
                        this,
                        std::ref(leftAbsImg),
                        std::ref(rightAbsImg),
                        std::ref(depthImgOut),
                        cv::Point2i(rows * i, rows * (i + 1)),
                        isMap,
                        isColor);
            }
            tasks[threads - 1] = std::thread(
                    &restructor::Restructor_CPU::thread_DepthColorMap,
                    this,
                    std::ref(leftAbsImg),
                    std::ref(rightAbsImg),
                    std::ref(depthImgOut),
                    cv::Point2i(rows * (threads - 1),
                                leftAbsImg.rows),
                    isMap,
                    isColor);
            for (int i = 0; i < threads; i++) {
                if (tasks[i].joinable()) {
                    tasks[i].join();
                }
            }
        }

        void Restructor_CPU::thread_DepthColorMap(
                const cv::Mat &leftAbsImg, const cv::Mat &righAbstImg,
                cv::Mat &depthImgOut, const cv::Point2i region, const bool isMap, const bool isColor) {
            const cv::Mat &Q = calibrationInfo.Q;
            const float f = Q.at<double>(2, 3);
            const float tx = -1.0 / Q.at<double>(3, 2);
            const float cxlr = Q.at<double>(3, 3) * tx;
            const float cx = -1.0 * Q.at<double>(0, 3);
            const float cy = -1.0 * Q.at<double>(1, 3);
            const float threshod = 0.3;
            const int rows = leftAbsImg.rows;
            const int cols = leftAbsImg.cols;
            //存放线性回归的实际值
            std::vector<cv::Point2f> linear_return = {
                    cv::Point2f(1, 1), cv::Point2f(1, 1),
                    cv::Point2f(1, 1), cv::Point2f(1, 1)};
            //存放回归拟合结果：0：dx，1：dy，2：x，3：y
            std::vector<float> result_Linear = {1, 1, 1, 1};
            //线性回归的斜率
            float a;
            //线性回归的截距
            float b;
            //搜寻到的匹配点列号
            int k;
            float minCost;
            float cost;
            float disparity;
            cv::Mat r1Inv = calibrationInfo.R1.inv();
            for (int i = region.x; i < region.y; i++) {
                const float *ptr_Left = leftAbsImg.ptr<float>(i);
                const float *ptr_Right = righAbstImg.ptr<float>(i);
                for (int j = 0; j < cols; j++) {
                    if (0.f >= ptr_Left[j]) {
                        continue;
                    }
                    minCost = FLT_MAX;
                    for (int d = minDisparity; d < maxDisparity; d++) {
                        if (j - d < 0 || j - d > cols - 1) {
                            continue;
                        }
                        cost = std::abs(ptr_Left[j] - ptr_Right[j - d]);
                        if (cost < minCost) {
                            k = d;
                            minCost = cost;
                        }
                    }
                    if (minCost > threshod) {
                        continue;
                    }
                    //以左边一位开始连续四点拟合
                    linear_return[0] = cv::Point2f(k + 1, ptr_Right[j - k - 1]);
                    linear_return[1] = cv::Point2f(k, ptr_Right[j - k]);
                    linear_return[2] = cv::Point2f(k - 1, ptr_Right[j - k + 1]);
                    linear_return[3] = cv::Point2f(k - 2, ptr_Right[j - k + 2]);
                    //线性拟合
                    cv::fitLine(linear_return, result_Linear, cv::DIST_L2, 0, 0.01, 0.01);
                    a = result_Linear[1] / result_Linear[0];
                    b = result_Linear[3] - a * result_Linear[2];
                    disparity = (ptr_Left[j] - b) / a;//获取到的亚像素匹配点

                    if (disparity < minDisparity || disparity > maxDisparity) {
                        continue;
                    }

                    cv::Mat recCameraPoints = (cv::Mat_<double>(3, 1) << -1.0 * tx * (j - cx) / (disparity - cxlr),
                                               -1.0 * tx * (i - cy) / (disparity - cxlr),
                                               -1.0 * tx * f / (disparity - cxlr));
                    cv::Mat cameraPoints, result;
                    if (isMap) 
                        cameraPoints = r1Inv * recCameraPoints;
                    else
                        cameraPoints = recCameraPoints;

                    if (isColor)
                        result = calibrationInfo.Rlc * cameraPoints + calibrationInfo.Tlc;
                    else
                        result = cameraPoints;

                    const float depth = result.at<double>(2, 0);

                    if (depth < minDepth || depth > maxDepth)
                        continue;

                    if (isMap) {
                        cv::Mat mapPicture = isColor ? calibrationInfo.M3 * result : calibrationInfo.M1 * result;
                        int x_maped = std::round(mapPicture.at<double>(0, 0) / mapPicture.at<double>(2, 0));
                        int y_maped = std::round(mapPicture.at<double>(1, 0) / mapPicture.at<double>(2, 0));

                        if ((0 > x_maped) || (y_maped > rows - 1) || (0 > y_maped) || (x_maped > cols - 1))
                            continue;

                        std::lock_guard<std::mutex> lock(mutexMap);
                        depthImgOut.ptr<float>(y_maped)[x_maped] = depth;
                    } 
                    else {
                        depthImgOut.ptr<float>(i)[j] = depth;
                    }
                }
            }
        }
    }// namespace restructor
}// namespace sl

