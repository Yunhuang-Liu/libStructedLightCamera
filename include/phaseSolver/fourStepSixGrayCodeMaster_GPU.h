/**
 * @file fourStepSixGrayCodeMaster_GPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU加速解相器(四步五位互补格雷码)
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef PHASESOLVER_FOURSTEPSIXGRAYCODEMASTER_GPU_H_
#define PHASESOLVER_FOURSTEPSIXGRAYCODEMASTER_GPU_H_

#include "phaseSolver.h"

/** @brief 结构光库 */
namespace sl {
    /** @brief 解相库 */
    namespace phaseSolver {
        /**
         * @brief 互补格雷码四步相移解码器(4+6 | CUDA)
         */
        class FourStepSixGrayCodeMaster_GPU : public PhaseSolver {
        public:
            /**
             * @brief 构造函数
             * @param block_ 输入，Block尺寸
             */
            FourStepSixGrayCodeMaster_GPU(const dim3 block_ = dim3(32, 8));
            /**
             * @brief 构造函数
             * @param imgs 输入，原始图片
             * @param block_ 输入，Block尺寸
             */
            FourStepSixGrayCodeMaster_GPU(std::vector<cv::Mat> &imgs,
                                          const dim3 block_ = dim3(32, 8));
            /**
             * @brief 析构函数
             */
            ~FourStepSixGrayCodeMaster_GPU();
            /**
             * @brief 解相
             *
             * @param unwrapImg 输入/输出，绝对相位图
             * @param pStream 输入，异步流
             */
            void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat> &unwrapImg,
                                   cv::cuda::Stream &pStream) override;
            /**
             * @brief 上传图片
             * @param imgs 输入，原始图片
             * @param pStream 输入，异步流
             */
            void changeSourceImg(std::vector<cv::Mat> &imgs,
                                 cv::cuda::Stream &stream) override;
            /**
             * @brief 上传原始图片
             * @param imgs 输入，设备端图片（CV_8UC1)
             */
            void changeSourceImg(std::vector<cv::cuda::GpuMat> &imgs) override;
            /**
             * @brief 获取纹理图片（灰度浮点型）
             * @param textureImg 输入/输出，纹理图片
             */
            void getTextureImg(std::vector<cv::cuda::GpuMat> &textureImg) override;

        protected:
            /**
             * @brief 上传图片
             * @param imgs 输入，原始图片
             */
            void changeSourceImg(std::vector<cv::Mat> &imgs) override;
        private:
            void getUnwrapPhaseImg(cv::Mat &) override {}
            void getWrapPhaseImg(cv::Mat &, cv::Mat &) override {}
            void getTextureImg(cv::Mat &textureImg) override {}
            /** \GPU图像  全部拍摄图片 0——3：相移 4——9：格雷 **/
            std::vector<cv::cuda::GpuMat> imgs_device;
            /** \阈值图像 **/
            cv::cuda::GpuMat averageImg_device;
            /** \调制图像 **/
            cv::cuda::GpuMat conditionImg_device;
            /** \包裹相位图像 **/
            cv::cuda::GpuMat wrapImg_device;
            /** \block尺寸 **/
            const dim3 block;
            /** \行数 **/
            int rows;
            /** \列数 **/
            int cols;
        };
    }// namespace phaseSolver
}// namespace sl
#endif // PHASESOLVER_FOURSTEPSIXGRAYCODEMASTER_GPU_H_
