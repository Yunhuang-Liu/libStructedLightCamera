/**
 * @file nStepNGrayCodeMaster_CPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  任意步任意位互补格雷码解相器
 * @version 0.1
 * @date 2022-10-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef PHASESOLVER_NSTEPNGRAYCODEMASTER_CPU_H_
#define PHASESOLVER_NSTEPNGRAYCODEMASTER_CPU_H_

#include <phaseSolver/phaseSolver.h>

#include <fstream>

/** @brief 结构光库 */
namespace sl {
    /** @brief 解相库 */
    namespace phaseSolver {
        /**
         * @brief N位互补格雷码N步相移解码器(多线程+SIMD)
         */
        class NStepNGrayCodeMaster_CPU : public PhaseSolver {
        public:
            /**
             * @brief           默认构造函数
             * 
             * @param shiftStep 输入，相移步数
             * @param threads   输入，线程数
             */
            NStepNGrayCodeMaster_CPU(const int shiftStep = 4, const int threads = 16);
            /**
             * @brief           构造函数
             * 
             * @param imgs      输入，图片
             * @param shiftStep 输入，相移步数
             * @param threads   输入，线程数
             */
            NStepNGrayCodeMaster_CPU(std::vector<cv::Mat> &imgs,
                                     const int shiftStep = 4,
                                     const int threads = 16);
            /**
             * @brief 析构函数
             */
            ~NStepNGrayCodeMaster_CPU();
            /**
             * @brief 解相
             * 
             * @param absolutePhaseImg 输入/输出，绝对相位
             */
            void getUnwrapPhaseImg(cv::Mat &absolutePhaseImg) override;
            /**
             * @brief 更改源图片
             * 
             * @param imgs 输入，原始图片
             */
            void changeSourceImg(std::vector<cv::Mat> &imgs) override;
            /**
             * @brief 获取包裹相位
             * 
             * @param wrapImg 输入，包裹相位
             * @param conditionImg 输入，调制图片
             */
            void getWrapPhaseImg(cv::Mat &wrapImg, cv::Mat &conditionImg) override;
            /**
             * @brief 获取纹理图片（灰度浮点型）
             * 
             * @param textureImg 输入/输出，纹理图片
             */
            void getTextureImg(cv::Mat &textureImg) override;

        protected:
            /**
             * @brief 获取包裹相位
             */
            void getWrapPhaseImg();

        private:
            /** \线程数 **/
            const int threads;
            #ifdef CUDA
            void changeSourceImg(std::vector<cv::Mat> &imgs,
                                 cv::cuda::Stream &stream) override {}
            void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat> &,
                                   cv::cuda::Stream &stream) override {}
            void getTextureImg(std::vector<cv::cuda::GpuMat> &textureImg) override {}
            void changeSourceImg(std::vector<cv::cuda::GpuMat> &imgs) override {}
            #endif
            /**
             * @brief 计算阈值图像
             */
            void caculateAverageImgs();
            /**
             * @brief 线程入口函数，多线程计算阈值图像
             *
             * @param imgs          输入，图片
             * @param step          输入，相移步数
             * @param wrapImg       输入，包裹图片
             * @param region        输入，图像操作区间
             * @param conditionImg  输出，调制图像
             * @param textureImg    输出，纹理图片
             */
            void dataInit_Thread_SIMD(const std::vector<cv::Mat> &imgs,
                                      const int step,
                                      const cv::Mat &wrapImg,
                                      const cv::Point2i region,
                                      cv::Mat &conditionImg,
                                      cv::Mat &textureImg);
            /**
             * @brief 线程入口函数，多线程解相
             *
             * @param imgs             输入，图片
             * @param shiftStep        输入，相移步数
             * @param conditionImg     输入，阈值图像
             * @param wrapImg          输入，包裹图像
             * @param region           输入，图像操作范围
             * @param absolutePhaseImg 输入输出，绝对相位图片
             */
            void mutiplyThreadUnwrap(const std::vector<cv::Mat> &imgs,
                                     const int shiftStep,
                                     const cv::Mat &conditionImg,
                                     const cv::Mat &wrapImg,
                                     const cv::Point2i region,
                                     cv::Mat &absolutePhaseImg);
            /**
             * @brief 计算包裹相位
             * 
             * @param imgs    输入，图片（前相移，后格雷码）
             * @param wrapImg 输入输出，包裹相位
             * @param step    输入，相移步数
             * @param threads 输入，线程数
             */
            void atan2M(const std::vector<cv::Mat> &imgs,
                        cv::Mat &wrapImg,
                        const int step = 4,
                        const int threads = 16);
            /**
             * @brief 线程入口函数，多线程计算包裹相位
             * 
             * @param imgs    输入，图片（前相移，后格雷）
             * @param step    输入，相移步数
             * @param region  输入，多线程区间
             * @param wrapImg 输入输出，包裹相位
             */
            void SIMD_WrapImg(const std::vector<cv::Mat> &imgs,
                              const int step,
                              const cv::Point2i &region,
                              cv::Mat &wrapImg);
            /** \全部拍摄图片 0——3：相移 4——8：格雷**/
            std::vector<cv::Mat> imgs;
            /** \阈值图像 **/
            cv::Mat averageImg;
            /** \调制阈值图像 **/
            cv::Mat conditionImg;
            /** \包裹相位图像 **/
            cv::Mat wrapImg;
            /** \相移步数 **/
            const int shiftStep;
        };
    }// namespace phaseSolver
}// namespace sl
#endif //PHASESOLVER_NSTEPNGRAYCODEMASTER_CPU_H_
