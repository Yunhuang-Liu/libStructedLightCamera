#include <restructor/restructor_GPU.h>

namespace sl {
    namespace restructor {
        Restructor_GPU::Restructor_GPU(
                const Info &calibrationInfo_, const int minDisparity_, const int maxDisparity_,
                const float minDepth_, const float maxDepth_, const dim3 block_) : calibrationInfo(calibrationInfo_),
                                                                                   minDisparity(minDisparity_),
                                                                                   maxDisparity(maxDisparity_),
                                                                                   minDepth(minDepth_),
                                                                                   maxDepth(maxDepth_),
                                                                                   block(block_) {
            cv::Mat Q_CV;
            cv::Mat R1_inv_CV;
            cv::Mat M1_CV;
            calibrationInfo_.M1.convertTo(M1_CV, CV_32FC1);
            calibrationInfo_.Q.convertTo(Q_CV, CV_32FC1);
            calibrationInfo_.R1.convertTo(R1_inv_CV, CV_32FC1);
            R1_inv_CV = R1_inv_CV.inv();
            cv::cv2eigen(Q_CV, Q);
            cv::cv2eigen(R1_inv_CV, R1_inv);
            cv::cv2eigen(M1_CV, M1);
            depthImg_device.resize(4);
            cv::Mat M3_CV;
            cv::Mat R_CV;
            cv::Mat T_CV;
            if (!calibrationInfo.M3.empty()) {
                calibrationInfo_.M3.convertTo(M3_CV, CV_32FC1);
                calibrationInfo_.R.convertTo(R_CV, CV_32FC1);
                calibrationInfo_.T.convertTo(T_CV, CV_32FC1);
                cv::cv2eigen(M3_CV, M3);
                cv::cv2eigen(R_CV, R);
                cv::cv2eigen(T_CV, T);
            }
            if (-500 == minDisparity && 500 == maxDisparity &&
                170 != minDepth && 220 != maxDepth) {
                float tx = -1.f / Q_CV.at<float>(3, 2);
                float crj = tx * Q_CV.at<float>(3, 3);
                float f = Q_CV.at<float>(2, 3);
                minDisparity = -tx * f / minDepth - crj;
                minDisparity = -tx * f / maxDepth - crj;
            }
        }

        Restructor_GPU::~Restructor_GPU() {
        }

        void Restructor_GPU::restruction(
                const cv::cuda::GpuMat &leftAbsImg, const cv::cuda::GpuMat &rightAbsImg,
                const int sysIndex, cv::cuda::Stream &stream, const bool isColor) {
            const int rows = leftAbsImg.rows;
            const int cols = leftAbsImg.cols;
            if (depthImg_device[sysIndex].empty()) {
                depthImg_device[sysIndex].create(rows, cols, CV_32FC1);
            } else {
                depthImg_device[sysIndex].setTo(0);
            }
            if (isColor) {
                getDepthColorMap(leftAbsImg, rightAbsImg, depthImg_device[sysIndex], stream);
            } else {
                getDepthMap(leftAbsImg, rightAbsImg, depthImg_device[sysIndex], stream);
            }
        }

        void Restructor_GPU::getDepthColorMap(
                const cv::cuda::GpuMat &leftImg, const cv::cuda::GpuMat &rightImg,
                cv::cuda::GpuMat &depthImg, cv::cuda::Stream &pStream) {
            const int rows = leftImg.rows;
            const int cols = leftImg.cols;
            restructor::cudaFunc::depthColorMap(leftImg, rightImg, rows, cols,
                                                    minDisparity, maxDisparity, minDepth, maxDepth, Q,
                                                    M3, R, T, R1_inv, depthImg, block, pStream);
        }

        void Restructor_GPU::getDepthMap(
                const cv::cuda::GpuMat &leftImg, const cv::cuda::GpuMat &rightImg,
                cv::cuda::GpuMat &depthImg, cv::cuda::Stream &pStream) {
            const int rows = leftImg.rows;
            const int cols = leftImg.cols;
            restructor::cudaFunc::depthMap(leftImg, rightImg, rows, cols,
                                               minDisparity, maxDisparity, minDepth, maxDepth, Q,
                                               M1, R1_inv, depthImg, block, pStream);
        }

        void Restructor_GPU::download(const int index, cv::cuda::GpuMat &depthImg) {
            depthImg = depthImg_device[index];
        }
    }// namespace restructor
}// namespace sl


