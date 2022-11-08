/**
 * @file wrapCreator_GPU.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  GPU加速包裹相位求解器
 * @version 0.1
 * @date 2022-08-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef WRAPCREATOR_WRAPCREATOR_GPU_H_
#define WRAPCREATOR_WRAPCREATOR_GPU_H_

#include <wrapCreator/wrapCreator.h>

/** @brief 结构光库 */
namespace sl {
    /** @brief 包裹生成库 */
    namespace wrapCreator {
        /** @brief GPU加速包裹相位求解器 */
        class WrapCreator_GPU : public WrapCreator {
        public:
            WrapCreator_GPU();
            ~WrapCreator_GPU();
            /**
             * @brief                   求取包裹相位
             * 
             * @param imgs              输入，相移图片
             * @param wrapImg           输入，包裹图片
             * @param conditionImg      输入，背景图片
             * @param cvStream          输入，非阻塞流
             * @param parameter         输入，加速参数
             */
            void getWrapImg(const std::vector<cv::cuda::GpuMat> &imgs,
                            cv::cuda::GpuMat &wrapImg,
                            cv::cuda::GpuMat &conditionImg, const bool isCounter = false,
                            cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                            const WrapParameter parameter = WrapParameter()) override;

        private:
            void getWrapImg(const std::vector<cv::Mat> &imgs,
                            cv::Mat &wrapImg,
                            cv::Mat &conditionImg, const bool isCounter = false,
                            const WrapParameter parameter = WrapParameter()) override{};
        };
    }// namespace wrapCreator
}// namespace sl
#endif// WRAPCREATOR_WRAPCREATOR_GPU_H_