#include <tool/tool.h>

namespace sl {
    namespace tool {
        void phaseHeightMapEigCoe(const cv::Mat& phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                  cv::Mat& depth, const int threads) {
            CV_Assert(intrinsic.type() == CV_64FC1);
            CV_Assert(coefficient.type() == CV_64FC1);
            depth = cv::Mat::zeros(phase.size(), CV_32FC1);

            const int rows = phase.rows;
            const int cols = phase.cols;
            const int patchRow = rows / threads;

            std::vector<std::thread> threadsPool(threads);
            for (int i = 0; i < threads - 1; ++i) {
                threadsPool[i] = std::thread(&phaseHeightMapEigCoeRegion, std::ref(phase), std::ref(intrinsic),
                                             std::ref(coefficient), rows * i, rows * (i + 1), std::ref(depth));
            }
            threadsPool[threads - 1] = std::thread(&phaseHeightMapEigCoeRegion, std::ref(phase), std::ref(intrinsic),
                                                   std::ref(coefficient), rows * (threads - 1), phase.rows, std::ref(depth));
        }

        void phaseHeightMapEigCoeRegion(const cv::Mat& phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                        const int rowBegin, const int rowEnd, cv::Mat& depth) {
            CV_Assert(intrinsic.type() == CV_64FC1);
            CV_Assert(coefficient.type() == CV_64FC1);
            CV_Assert(depth.type() == CV_32FC1);
            CV_Assert(!depth.empty());

            cv::Mat intrinsicInvCV;
            intrinsic.convertTo(intrinsicInvCV, CV_32FC1);
            intrinsicInvCV = intrinsicInvCV.inv();

            Eigen::Matrix3f mapL, intrinsicInv;
            Eigen::Vector3f mapR, cameraPoint, imgPoint;
            cv::cv2eigen(intrinsicInvCV, intrinsicInv);
            mapL << intrinsicInvCV.ptr<float>(0)[0], 0, 0,
                    0, intrinsicInvCV.ptr<float>(1)[1], 0,
                    0, 0, 0;
            mapR << 0, 0, 0;


            for (int i = rowBegin; i < rowEnd; ++i) {
                const float *ptrPhase = phase.ptr<float>(i);
                float *ptrDepth = depth.ptr<float>(i);
                for (int j = 0; j < phase.cols; ++j) {
                    mapL(0, 2) = intrinsic.ptr<double>(0)[2] - j;
                    mapL(1, 2) = intrinsic.ptr<double>(1)[2] - i;
                    mapL(2, 0) = coefficient.ptr<double>(0)[0] - coefficient.ptr<double>(0)[4] * ptrPhase[j];
                    mapL(2, 1) = coefficient.ptr<double>(0)[1] - coefficient.ptr<double>(0)[5] * ptrPhase[j];
                    mapL(2, 2) = coefficient.ptr<double>(0)[2] - coefficient.ptr<double>(0)[6] * ptrPhase[j];

                    mapR(2, 0) = coefficient.ptr<double>(0)[7] * ptrPhase[j] - coefficient.ptr<double>(0)[3];
                    cameraPoint = mapL.inverse() * mapR;

                    ptrDepth[j] = cameraPoint.z();
                    /*
                    imgPoint = intrinsicInv * cameraPoint;
                    int locX = imgPoint(0, 0) / imgPoint(2, 0);
                    int locY = imgPoint(0, 0) / imgPoint(2, 0);

                    if (locX > 0 && locX < depth.rows && locY > 0 && locY < depth.cols) {
                        depth.ptr<float>(locY)[locX] = cameraPoint.z();
                    }
                    */
                }
            }
        }
    }
}