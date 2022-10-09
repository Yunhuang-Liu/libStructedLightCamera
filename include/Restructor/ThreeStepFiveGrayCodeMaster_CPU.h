/**
 * @file ThreeStepFiveGrayCodeMaster_CPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  三步四位互补格雷码CPU加速解相器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef RESTRUCTOR_THREESTEPFIVEGRAYCODEMASTER_CPU_H
#define RESTRUCTOR_THREESTEPFIVEGRAYCODEMASTER_CPU_H

#include <Restructor/PhaseSolver.h>
#include <fstream>

/** @brief 结构光库 */
namespace SL {
    /** @brief 重建库 */
    namespace PhaseSolverType {
        /**
         * @brief 互补格雷码四步相移解码器(3+5 | 多线程+SIMD)
         */
        class ThreeStepFiveGrayCodeMaster_CPU : public PhaseSolver {
        public:
            /**
             * @brief 默认构造函数
             * 
             * @param threads 输入，线程数
             */
            ThreeStepFiveGrayCodeMaster_CPU(const int threads = 16);
            /**
             * @brief 构造函数
             * 
             * @param imgs 输入，原始图片
             * @param threads 输入，线程数
             */
            ThreeStepFiveGrayCodeMaster_CPU(std::vector<cv::Mat> &imgs, const int threads = 16);
            /**
             * @brief 析构函数
             */
            ~ThreeStepFiveGrayCodeMaster_CPU();
            /**
             * @brief 获取绝对相位图片
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
             * @brief 获取纹理图片
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
#ifdef CUDA
            void changeSourceImg(std::vector<cv::Mat> &, cv::cuda::Stream &) override {}
            void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat> &, cv::cuda::Stream &) override {}
            void getTextureImg(std::vector<cv::cuda::GpuMat> &textureImg) override {}
            void changeSourceImg(std::vector<cv::cuda::GpuMat> &imgs) override {}
#endif
            /** \阈值图像 **/
            cv::Mat averageImg;
            /** \调制阈值图像 **/
            cv::Mat conditionImg;
            /** \包裹相位图像 **/
            cv::Mat wrapImg;
            /**
             * @brief 计算反正切夹角值(-pi ~ pi)
             *
             * @param firstStep 输入，第一步图像
             * @param secondStep 输入，第二步图像
             * @param thirdStep 输入，第三步图像
             * @param wrapImg 输出，包裹相位图片
             */
            void atan3M(const cv::Mat &firstStep, const cv::Mat &secondStep, const cv::Mat &thirdStep, cv::Mat &wrapImg, const int threads = 16);
            /**
             * @brief 计算包裹相位
             *
             * @param firstStep 输入，第一步图像
             * @param secondStep 输入，第二步图像
             * @param thirdStep 输入，第三步图像
             * @param region 输入，多线程区间
             * @param wrapImg 输出，包裹相位图片
             */
            void SIMD_WrapImg(const cv::Mat &firstStep, const cv::Mat &secondStep, const cv::Mat &thirdStep, const cv::Point2i &region, cv::Mat &wrapImg);
            /**
             * @brief 计算阈值图像
             *
             */
            void caculateAverageImgs();
            /**
             * @brief 线程入口函数，多线程计算阈值图像
             *
             * @param rows 输入，行数
             * @param cols 输入，列数
             * @param region 输入，图像操作区间
             */
            void dataInit_Thread_SIMD(const int rows, const int cols, const cv::Point2i region);
            /**
             * @brief 线程入口函数，多线程解相
             *
             * @param rows 输入，行数
             * @param cols 输入，列数
             * @param region 输入，图像操作范围
             * @param absolutePhaseImg 输入输出，绝对相位图片
             */
            void mutiplyThreadUnwrap(const int rows, const int cols, const cv::Point2i region, cv::Mat &absolutePhaseImg);
            /** \全部拍摄图片 0——3：相移 4——9：格雷 10？颜色 **/
            std::vector<cv::Mat> &imgs;
            /** \线程数 **/
            const int threads;
        };
    }// namespace PhaseSolverType
}// namespace SL
#endif // RESTRUCTOR_THREESTEPFIVEGRAYCODEMASTER_CPU_H
