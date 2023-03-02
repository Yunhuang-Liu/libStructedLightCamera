#include "../../include/tool/tool.h"

namespace sl {
    namespace tool {
        void phaseHeightMapEigCoe(const cv::Mat &phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                  const float minDepth, const float maxDepth,
                                  cv::Mat &depth, const int threads) {
            CV_Assert(intrinsic.type() == CV_64FC1);
            CV_Assert(coefficient.type() == CV_64FC1);
            depth = cv::Mat::zeros(phase.size(), CV_32FC1);

            const int rows = phase.rows;
            const int cols = phase.cols;
            const int patchRow = rows / threads;

            std::vector<std::thread> threadsPool(threads);
            for (int i = 0; i < threads - 1; ++i) {
                threadsPool[i] = std::thread(&phaseHeightMapEigCoeRegion, std::ref(phase), std::ref(intrinsic),
                                             std::ref(coefficient), minDepth, maxDepth,
                                             patchRow * i, patchRow * (i + 1), std::ref(depth));
            }
            threadsPool[threads - 1] = std::thread(&phaseHeightMapEigCoeRegion, std::ref(phase), std::ref(intrinsic),
                                                   std::ref(coefficient), minDepth, maxDepth,
                                                   patchRow * (threads - 1), phase.rows, std::ref(depth));
            for (auto &thread: threadsPool)
                thread.join();
        }

        void phaseHeightMapEigCoeRegion(const cv::Mat &phase, const cv::Mat &intrinsic, const cv::Mat &coefficient,
                                        const float minDepth, const float maxDepth,
                                        const int rowBegin, const int rowEnd, cv::Mat &depth) {
            CV_Assert(intrinsic.type() == CV_64FC1);
            CV_Assert(coefficient.type() == CV_64FC1);
            CV_Assert(depth.type() == CV_32FC1);
            CV_Assert(!depth.empty());

            cv::Mat intrinsicFT;
            intrinsic.convertTo(intrinsicFT, CV_32FC1);

            Eigen::Matrix3f mapL;
            Eigen::Vector3f mapR, cameraPoint, imgPoint;
            mapL << intrinsicFT.ptr<float>(0)[0], 0, 0,
                    0, intrinsicFT.ptr<float>(1)[1], 0,
                    0, 0, 0;
            mapR << 0, 0, 0;


            for (int i = rowBegin; i < rowEnd; ++i) {
                const float *ptrPhase = phase.ptr<float>(i);
                float *ptrDepth = depth.ptr<float>(i);
                for (int j = 0; j < phase.cols; ++j) {
                    if (ptrPhase[j] == -5.f) {
                        ptrDepth[j] = 0.f;
                        continue;
                    }
                    mapL(0, 2) = intrinsic.ptr<double>(0)[2] - j;
                    mapL(1, 2) = intrinsic.ptr<double>(1)[2] - i;
                    mapL(2, 0) = coefficient.ptr<double>(0)[0] - coefficient.ptr<double>(4)[0] * ptrPhase[j];
                    mapL(2, 1) = coefficient.ptr<double>(1)[0] - coefficient.ptr<double>(5)[0] * ptrPhase[j];
                    mapL(2, 2) = coefficient.ptr<double>(2)[0] - coefficient.ptr<double>(6)[0] * ptrPhase[j];

                    mapR(2, 0) = coefficient.ptr<double>(7)[0] * ptrPhase[j] - coefficient.ptr<double>(3)[0];
                    cameraPoint = mapL.inverse() * mapR;

                    ptrDepth[j] = cameraPoint.z();
                }
            }
        }

        void averageTexture(std::vector<cv::Mat> &imgs, cv::Mat &texture, const int phaseShiftStep, const int threads) {
            CV_Assert(imgs.size() >= phaseShiftStep);
            CV_Assert(imgs[0].type() == CV_8UC1 || imgs[0].type() == CV_8UC3);

            const bool isColor = imgs[0].type() == CV_8UC3;
            if (isColor)
                texture = cv::Mat(imgs[0].size(), imgs[0].type(), cv::Scalar(0, 0, 0));
            else
                texture = cv::Mat(imgs[0].size(), imgs[0].type(), cv::Scalar(0));

            const int perRow = texture.rows / threads;
            std::vector<std::thread> threadsPool(threads);

            if (isColor) {
                for (int i = 0; i < threads - 1; ++i) {
                    threadsPool[i] = std::thread(&sl::tool::averageTextureRegionColor, std::ref(imgs), std::ref(texture), phaseShiftStep,
                                                 perRow * i, perRow * (i + 1));
                }
                threadsPool[threads - 1] = std::thread(&sl::tool::averageTextureRegionColor, std::ref(imgs), std::ref(texture), phaseShiftStep,
                                                       perRow * (threads - 1), texture.rows);
            } else {
                for (int i = 0; i < threads - 1; ++i) {
                    threadsPool[i] = std::thread(&sl::tool::averageTextureRegionGrey, std::ref(imgs), std::ref(texture), phaseShiftStep,
                                                 perRow * i, perRow * (i + 1));
                }
                threadsPool[threads - 1] = std::thread(&sl::tool::averageTextureRegionGrey, std::ref(imgs), std::ref(texture), phaseShiftStep,
                                                       perRow * (threads - 1), texture.rows);
            }

            for (auto &thread: threadsPool)
                thread.join();
        }

        void averageTextureRegionGrey(std::vector<cv::Mat> &imgs, cv::Mat &texture, const int phaseShiftStep,
                                      const int rowBegin, const int rowEnd) {
            const int rows = texture.rows;
            const int cols = texture.cols;

            std::vector<uchar *> ptrImgs(phaseShiftStep);
            for (int i = rowBegin; i < rowEnd; ++i) {
                for (int p = 0; p < phaseShiftStep; ++p)
                    ptrImgs[p] = imgs[p].ptr<uchar>(i);
                uchar *ptrTexture = texture.ptr<uchar>(i);
                for (int j = 0; j < cols; ++j) {
                    float imgVal = 0.f;
                    for (int p = 0; p < phaseShiftStep; ++p)
                        imgVal += ptrImgs[p][j];
                    imgVal /= phaseShiftStep;
                    ptrTexture[j] = static_cast<uchar>(imgVal);
                }
            }
        }

        void averageTextureRegionColor(std::vector<cv::Mat> &imgs, cv::Mat &texture, const int phaseShiftStep,
                                       const int rowBegin, const int rowEnd) {
            const int rows = texture.rows;
            const int cols = texture.cols;

            std::vector<cv::Vec3b *> ptrImgs(phaseShiftStep);
            for (int i = rowBegin; i < rowEnd; ++i) {
                for (int p = 0; p < phaseShiftStep; ++p)
                    ptrImgs[p] = imgs[p].ptr<cv::Vec3b>(i);
                cv::Vec3b *ptrTexture = texture.ptr<cv::Vec3b>(i);
                for (int j = 0; j < cols; ++j) {
                    cv::Vec3f imgVal = 0.f;
                    for (int p = 0; p < phaseShiftStep; ++p)
                        imgVal += ptrImgs[p][j];
                    imgVal /= phaseShiftStep;
                    ptrTexture[j] = static_cast<cv::Vec3b>(imgVal);
                }
            }
        }

        void reverseMappingTexture(cv::Mat &depth, cv::Mat &textureIn, const Info &info, cv::Mat &textureAlign, const int threads) {
            CV_Assert(!depth.empty() && !textureIn.empty() && !info.M1.empty() && !info.M3.empty() && !info.Rlc.empty() && !info.Tlc.empty());
            CV_Assert(depth.type() == CV_32FC1 || textureIn.type() == CV_8UC3);

            textureAlign = cv::Mat(textureIn.size(), textureIn.type(), cv::Scalar(0, 0, 0));

            const int perRow = depth.rows / threads;
            std::vector<std::thread> threadsPool(threads);

            for (int i = 0; i < threads - 1; ++i) {
                threadsPool[i] = std::thread(&sl::tool::reverseMappingTextureRegion, std::ref(depth), std::ref(textureIn), std::ref(info), std::ref(textureAlign),
                                             perRow * i, perRow * (i + 1));
            }
            threadsPool[threads - 1] = std::thread(&sl::tool::reverseMappingTextureRegion, std::ref(depth), std::ref(textureIn), std::ref(info), std::ref(textureAlign),
                                                   perRow * (threads - 1), depth.rows);

            for (auto &thread: threadsPool)
                thread.join();
        }

        void reverseMappingTextureRegion(cv::Mat &depth, cv::Mat &textureIn, const Info &info, cv::Mat &textureAlign, const int rowBegin, const int rowEnd) {
            const int rows = depth.rows;
            const int cols = depth.cols;

            const cv::Mat M1Inv = info.M1.inv();
            const cv::Mat M3 = info.M3;
            for (int i = rowBegin; i < rowEnd; ++i) {
                auto ptrDepth = depth.ptr<float>(i);
                auto ptrTextureAlign = textureAlign.ptr<cv::Vec3b>(i);
                for (int j = 0; j < cols; ++j) {
                    if (ptrDepth[j] == 0.f)
                        continue;

                    cv::Mat colorCameraPoint = info.Rlc * M1Inv * (cv::Mat_<double>(3, 1) << j * ptrDepth[j], i * ptrDepth[j], ptrDepth[j]) + info.Tlc;
                    cv::Mat imgPoint = M3 * colorCameraPoint;
                    const int xMapped = imgPoint.at<double>(0, 0) / imgPoint.at<double>(2, 0);
                    const int yMapped = imgPoint.at<double>(1, 0) / imgPoint.at<double>(2, 0);

                    if (xMapped < 0 || xMapped > cols - 1 || yMapped < 0 || yMapped > rows - 1) {
                        ptrTextureAlign[j] = cv::Vec3b(0, 0, 0);
                        continue;
                    }

                    ptrTextureAlign[j] = textureIn.ptr<cv::Vec3b>(yMapped)[xMapped];
                }
            }
        }

        void reverseMappingRefineNew(const cv::Mat& phase, const cv::Mat& depth,
            const cv::Mat& wrap, const cv::Mat& condition,
            const Eigen::Matrix3f& intrinsicInvD, const Eigen::Matrix3f& intrinsicR,
            const Eigen::Matrix3f& RDtoR, const Eigen::Vector3f& TDtoR,
            const Eigen::Matrix4f& PL, const Eigen::Matrix4f& PR, const float threshod,
            const cv::cuda::GpuMat& epilineA, const cv::cuda::GpuMat& epilineB, const cv::cuda::GpuMat& epilineC,
            cv::cuda::GpuMat& depthRefine, const int threads = 16) {

        }

        void reverseMappingRefineRegionNew(const cv::Mat& phase, const cv::Mat& depth,
            const cv::Mat& wrap, const cv::Mat& condition,
            const Eigen::Matrix3f& intrinsicInvD, const Eigen::Matrix3f& intrinsicR,
            const Eigen::Matrix3f& RDtoR, const Eigen::Vector3f& TDtoR,
            const Eigen::Matrix4f& PL, const Eigen::Matrix4f& PR, const float threshod,
            const cv::Mat& epilineA, const cv::Mat& epilineB, const cv::Mat& epilineC,
            const int rowBegin, const int rowEnd,
            cv::Mat& depthRefine) {

        }
    }
}
