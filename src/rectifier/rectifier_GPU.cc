#include "../../include/rectifier/rectifier_GPU.h"

namespace sl {
    namespace rectifier {
        Rectifier_GPU::Rectifier_GPU() {
        }

        Rectifier_GPU::Rectifier_GPU(const tool::Info &info) {
            m_imgSize = cv::Size(info.S.ptr<double>(0)[0], info.S.ptr<double>(1)[0]);
            cv::Mat m_map_Lx_, m_map_Ly_, m_map_Rx_, m_map_Ry_;
            cv::initUndistortRectifyMap(info.M1, info.D1, info.R1, info.P1, m_imgSize, CV_32FC1, m_map_Lx_, m_map_Ly_);
            cv::initUndistortRectifyMap(info.M2, info.D2, info.R2, info.P2, m_imgSize, CV_32FC1, m_map_Rx_, m_map_Ry_);
            m_map_Lx.upload(m_map_Lx_);
            m_map_Ly.upload(m_map_Ly_);
            m_map_Rx.upload(m_map_Rx_);
            m_map_Ry.upload(m_map_Ry_);
        }

        void Rectifier_GPU::initialize(const tool::Info &info) {
            m_imgSize = cv::Size(info.S.ptr<double>(0)[0], info.S.ptr<double>(1)[0]);
            cv::Mat m_map_Lx_, m_map_Ly_, m_map_Rx_, m_map_Ry_;
            cv::initUndistortRectifyMap(info.M1, info.D1, info.R1, info.P1, m_imgSize, CV_32FC1, m_map_Lx_, m_map_Ly_);
            cv::initUndistortRectifyMap(info.M2, info.D2, info.R2, info.P2, m_imgSize, CV_32FC1, m_map_Rx_, m_map_Ry_);
            m_map_Lx.upload(m_map_Lx_);
            m_map_Ly.upload(m_map_Ly_);
            m_map_Rx.upload(m_map_Rx_);
            m_map_Ry.upload(m_map_Ry_);
        }

        void Rectifier_GPU::remapImg(cv::Mat &imgInput, cv::cuda::GpuMat &imgOutput,
                                     cv::cuda::Stream &cvStream, const bool isLeft) {
            cv::cuda::GpuMat imgDev;
            imgDev.upload(imgInput, cvStream);
            if (isLeft)
                cv::cuda::remap(imgDev, imgOutput, m_map_Lx, m_map_Ly, cv::INTER_LINEAR,
                                0, cv::Scalar(), cvStream);
            else
                cv::cuda::remap(imgDev, imgOutput, m_map_Rx, m_map_Ry, cv::INTER_LINEAR,
                                0, cv::Scalar(), cvStream);
        }
    }// namespace rectifier
}// namespace sl
