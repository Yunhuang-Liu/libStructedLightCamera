#ifndef RESTRUCTOR_RECTIFIER_CPU_H_
#define RESTRUCTOR_RECTIFIER_CPU_H_

#include <Restructor/Rectifier.h>

/** @brief 结构光库 */
namespace sl {
    /** @brief 极线校正库 */
    namespace rectifier {
        /** @brief CPU极线校正器 */
        class Rectifier_CPU : public Rectifier {
        public:
            Rectifier_CPU();
            /**
             * @brief           使用标定参数构造极线校正器
             * 
             * @param info      输入，相机标定参数
             */
            Rectifier_CPU(const Info &info);
            /**
             * @brief           使用标定参数初始化极线校正器
             * 
             * @param info      输入，相机标定参数
             */
            void initialize(const Info &info);
            /**
             * @brief           对图片进行极线校正
             * 
             * @param imgInput  输入，图片
             * @param imgOutput 输出，图片
             * @param isLeft    输入，是否进行左相机校正：true 左，false 右
             */
            void remapImg(cv::Mat &imgInput, cv::Mat &imgOutput, const bool isLeft = true);
        private:
#ifdef CUDA
            void remapImg(cv::Mat &imgInput, cv::cuda::GpuMat &imgOutput,
                          cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(), const bool isLeft = true) {}
#endif      
            //图像尺寸
            cv::Size m_imgSize;
            //左相机X方向映射表
            cv::Mat m_map_Lx;
            //左相机Y方向映射表
            cv::Mat m_map_Ly;
            //右相机X方向映射表
            cv::Mat m_map_Rx;
            //右相机Y方向映射表
            cv::Mat m_map_Ry;
        };
    }// namespace rectifier
}// namespace sl
#endif // RESTRUCTOR_RECTIFIER_CPU_H_