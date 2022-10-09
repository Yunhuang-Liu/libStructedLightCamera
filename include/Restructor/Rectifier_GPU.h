#ifndef RESTRUCTOR_RECTIFIER_GPU_H_
#define RESTRUCTOR_RECTIFIER_GPU_H_

#include <Restructor/Rectifier.h>

/** @brief 结构光库 */
namespace SL {
    /** @brief 极线校正库 */
    namespace Rectify {
        /** @brief GPU极线校正器 */
        class Rectifier_GPU : public Rectifier {
        public:
            Rectifier_GPU();
            /**
             * @brief           使用标定参数构造极线校正器
             * 
             * @param info      输入，相机标定参数
             */
            Rectifier_GPU(const Info &info);
            /**
             * @brief           使用标定参数初始化极线校正器
             * 
             * @param info      输入，相机标定参数
             */
            void initialize(const Info &info);
            /**
             * @brief                对图片进行极线校正
             * 
             * @param imgInput       输入，图片
             * @param imgInput       输出，校正后图片
             * @param cvStream       输入，异步流
             * @param isLeft         输入，是否进行左相机校正：true 左，false 右
             */
            void remapImg(cv::Mat &imgInput, cv::cuda::GpuMat &imgOutput,
                          cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(), const bool isLeft = true);

        private:
            void remapImg(cv::Mat &imgInput, cv::Mat &imgOutput, const bool isLeft = true) {}
            //图像大小
            cv::Size m_imgSize;
            //左相机X方向映射表
            cv::cuda::GpuMat m_map_Lx;
            //右相机Y方向映射表
            cv::cuda::GpuMat m_map_Ly;
            //左相机X方向映射表
            cv::cuda::GpuMat m_map_Rx;
            //右相机Y方向映射表
            cv::cuda::GpuMat m_map_Ry;
        };
    }// namespace Rectify
}// namespace SL
#endif // RESTRUCTOR_RECTIFIER_GPU_H_