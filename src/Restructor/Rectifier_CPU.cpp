#include <Restructor/Rectifier_CPU.h>

namespace SL {
    namespace Rectify {
        Rectifier_CPU::Rectifier_CPU() {
        }

        Rectifier_CPU::Rectifier_CPU(const Info &info) {
            m_imgSize = cv::Size(info.S.ptr<double>(0)[0], info.S.ptr<double>(1)[0]);
            cv::initUndistortRectifyMap(info.M1, info.D1, info.R1, info.P1, m_imgSize, CV_16SC2, m_map_Lx, m_map_Ly);
            cv::initUndistortRectifyMap(info.M2, info.D2, info.R2, info.P2, m_imgSize, CV_16SC2, m_map_Rx, m_map_Ry);
        }

        void Rectifier_CPU::initialize(const Info &info) {
            m_imgSize = cv::Size(info.S.ptr<double>(0)[0], info.S.ptr<double>(1)[0]);
            cv::initUndistortRectifyMap(info.M1, info.D1, info.R1, info.P1, m_imgSize, CV_16SC2, m_map_Lx, m_map_Ly);
            cv::initUndistortRectifyMap(info.M2, info.D2, info.R2, info.P2, m_imgSize, CV_16SC2, m_map_Rx, m_map_Ry);
        }

        void Rectifier_CPU::remapImg(cv::Mat &imgInput, cv::Mat &imgOutput, const bool isLeft) {
            if (isLeft)
                cv::remap(imgInput, imgOutput, m_map_Lx, m_map_Ly, cv::INTER_LINEAR);
            else
                cv::remap(imgInput, imgOutput, m_map_Rx, m_map_Ry, cv::INTER_LINEAR);
        }
    }// namespace Rectify
}// namespace SL