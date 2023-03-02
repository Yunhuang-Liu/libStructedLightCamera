/**
 * @file fourStepRefPlainMaster_GPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  四步参考平面GPU加速解相器
 * @version 0.1
 * @date 2022-10-09
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef PHASESOLVER_FOURSTEPREFPLAINMASTER_GPU_H_
#define PHASESOLVER_FOURSTEPREFPLAINMASTER_GPU_H_

#include "phaseSolver.h"
#include "../wrapCreator/wrapCreator_GPU.h"

/** @brief 结构光库 */
namespace sl {
    /** @brief 解相库 */
    namespace phaseSolver {
        /**
         * @brief 四步参考平面解相器
         *
         * @note    CUDA加速版本
         * @warning 由于包裹求解器目前只支持三步和四步相移，因此请勿使用其它步相移算法
         */
        class FourStepRefPlainMaster_GPU : public PhaseSolver {
        public:
            /**
             * @brief 构造函数
             * @param block_ 输入，Block尺寸
             */
            FourStepRefPlainMaster_GPU(const cv::Mat refPlain, const bool isFarestMode, const dim3 block_ = dim3(32, 8));
            /**
             * @brief 析构函数
             */
            ~FourStepRefPlainMaster_GPU();
            /**
             * @brief 解相
             *
             * @param unwrapImg 输入/输出，绝对相位图
             * @param pStream 输入，异步流
             */
            void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat> &unwrapImg,
                                   cv::cuda::Stream &pStream) override;
            /**
             * @brief 上传原始图片
             *
             * @param imgs 输入，设备端图片
             * @note 图片类型为CV_8UC1
             */
            void changeSourceImg(std::vector<cv::cuda::GpuMat> &imgs) override;
            /**
             * @brief 获取纹理图片（灰度浮点型）
             *
             * @param textureImg 输入/输出，纹理图片
             */
            void getTextureImg(std::vector<cv::cuda::GpuMat> &textureImg) override;
            /**
             * @brief 设置连续帧逆序模式
             *
             * @param isCounter 输入，是否逆序
             */
            inline void setCounterMode(const bool isCounter) {
                m_isCounter = isCounter;
            }
        private:
            void changeSourceImg(std::vector<cv::Mat> &imgs) override {};
            void changeSourceImg(std::vector<cv::Mat> &imgs,
                                 cv::cuda::Stream &stream) override {};
            void getUnwrapPhaseImg(cv::Mat &) override {}
            void getWrapPhaseImg(cv::Mat &, cv::Mat &) override {}
            void getTextureImg(cv::Mat &textureImg) override {}
            /** \包裹求解器 **/
            std::unique_ptr<sl::wrapCreator::WrapCreator> wrapCreator;
            /** \GPU图像  全部拍摄图片**/
            std::vector<cv::cuda::GpuMat> m_imgs;
            /** \调制图像 **/
            cv::cuda::GpuMat m_conditionImg;
            /** \包裹相位图像 **/
            cv::cuda::GpuMat m_wrapImg;
            /** \参考平面图像 **/
            const cv::cuda::GpuMat m_refPlainImg;
            /** \block尺寸 **/
            const dim3 m_block;
            /** \是否连续帧逆序 **/
            bool m_isCounter;
            /** \是否为最远参考平面 **/
            bool m_isFarestPlain;
        };
    }// namespace phaseSolver
}// namespace sl
#endif// PHASESOLVER_FOURSTEPREFPLAINMASTER_GPU_H_
