#include <Restructor/Restructor_GPU.h>

namespace RestructorType {
    Restructor_GPU::Restructor_GPU(const Info& calibrationInfo_, const int minDisparity_, const int maxDisparity_,
        const float minDepth_, const float maxDepth_, const dim3 block_) :
        calibrationInfo(calibrationInfo_),
        minDisparity(minDisparity_),
        maxDisparity(maxDisparity_),
        minDepth(minDepth_),
        maxDepth(maxDepth_),
        block(block_){
        cv::Mat Q_CV;
        cv::Mat M3_CV;
        cv::Mat R_CV;
        cv::Mat T_CV;
        cv::Mat R1_inv_CV;
        calibrationInfo_.Q.convertTo(Q_CV, CV_32FC1);
        calibrationInfo_.M3.convertTo(M3_CV, CV_32FC1);
        calibrationInfo_.R.convertTo(R_CV, CV_32FC1);
        calibrationInfo_.T.convertTo(T_CV, CV_32FC1);
        calibrationInfo_.R1.convertTo(R1_inv_CV, CV_32FC1);
        R1_inv_CV = R1_inv_CV.inv();
        cv::cv2eigen(Q_CV,Q);
        cv::cv2eigen(M3_CV,M3);
        cv::cv2eigen(R_CV,R);
        cv::cv2eigen(T_CV,T);
        cv::cv2eigen(R1_inv_CV,R1_inv);
        depthImg_device.resize(4);
        colorImg_device.resize(4);
    }

    Restructor_GPU::~Restructor_GPU() {
    }

    void Restructor_GPU::restruction(const cv::cuda::GpuMat& leftAbsImg, const cv::cuda::GpuMat& rightAbsImg, const cv::Mat& colorImg,
                                     const int sysIndex, cv::cuda::Stream& stream) {
        const int rows =leftAbsImg.rows;
        const int cols = leftAbsImg.cols;
        if (depthImg_device[sysIndex].empty()) {
            depthImg_device[sysIndex].create(rows, cols, CV_16UC1);
            colorImg_device[sysIndex].create(rows, cols, CV_8UC3);
        }
        else {
            depthImg_device[sysIndex].setTo(0);
            colorImg_device[sysIndex].setTo(cv::Vec3b(0, 0, 0));
        }
        cv::cuda::GpuMat colorImg_srcDev(colorImg);
        getDepthColorMap(leftAbsImg,rightAbsImg, colorImg_srcDev, depthImg_device[sysIndex], colorImg_device[sysIndex],stream);
    }

    void Restructor_GPU::getDepthColorMap(const cv::cuda::GpuMat& leftImg, const cv::cuda::GpuMat& rightImg, const cv::cuda::GpuMat& colorImg, cv::cuda::GpuMat& depthImg, cv::cuda::GpuMat& mapColorImg, cv::cuda::Stream& pStream) {
        const int rows =leftImg.rows;
        const int cols = leftImg.cols;
        RestructorType::cudaFunc::depthColorMap(leftImg, rightImg, colorImg, rows, cols,
            minDisparity, maxDisparity, minDepth, maxDepth, Q,
            M3, R, T, R1_inv, depthImg,mapColorImg, block, pStream);
    }

    void Restructor_GPU::download(const int index, cv::cuda::GpuMat& depthImg, cv::cuda::GpuMat& colorImg) {
        depthImg = depthImg_device[index];
        colorImg = colorImg_device[index];
    }
}


