/**
 * @file FourStepSixGrayCodeMaster_CPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  四步五位互补格雷码CPU加速解相器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef FourStepSixGrayCodeMaster_CPU_H
#define FourStepSixGrayCodeMaster_CPU_H
#include "./PhaseSolver.h"
#include <fstream>

namespace PhaseSolverType {
    /**
     * @brief 互补格雷码四步相移解码器(4+6 | 多线程+SIMD)
     *
     */
    class FourStepSixGrayCodeMaster_CPU : public PhaseSolver{
        public:
            /**
            * @brief 默认构造函数
            * @param refImg 输入，参考相位用于虚拟平面扩展格雷码，若无需，则将其置为空Mat
            * @param threads 输入，线程数
            */
            FourStepSixGrayCodeMaster_CPU(const cv::Mat& refImg = cv::Mat(0, 0, CV_32FC1), const int threads = 16);
            /**
             * @brief 构造函数
             * @param imgs 输入，图片
             * @param refImg 输入，参考相位用于虚拟平面扩展格雷码，若无需，则将其置为空Mat
             * @param threads 输入，线程数
             */
            FourStepSixGrayCodeMaster_CPU(std::vector<cv::Mat>& imgs, const cv::Mat& refImg = cv::Mat(0,0,CV_32FC1), const int threads = 16);
            /**
             * @brief 析构函数
             */
            ~FourStepSixGrayCodeMaster_CPU();
            /**
             * @brief 解相
             * @param absolutePhaseImg 输入/输出，绝对相位
             */
            void getUnwrapPhaseImg(cv::Mat& absolutePhaseImg) override;
            /**
             * @brief 更改源图片
             * @param imgs 输入，原始图片
             */
            void changeSourceImg(std::vector<cv::Mat>& imgs) override;
            /**
             * @brief 获取包裹相位
             * @param wrapImg 输入，包裹相位
             * @param conditionImg 输入，调制图片
             */
            void getWrapPhaseImg(cv::Mat& wrapImg, cv::Mat& conditionImg) override;
            /**
             * @brief 获取纹理图片（灰度浮点型）
             * @param textureImg 输入/输出，纹理图片
             */
            void getTextureImg(cv::Mat& textureImg) override;
        protected:
            /**
             * @brief 获取包裹相位
             */
            void getWrapPhaseImg();
        private:
            #ifdef CUDA
            /** \继承方法 **/
            void changeSourceImg(std::vector<cv::Mat>& imgs, cv::cuda::Stream& stream) override{}
            void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>&, cv::cuda::Stream& stream) override{}
            void getTextureImg(std::vector<cv::cuda::GpuMat>& textureImg) override {}
            void changeSourceImg(std::vector<cv::cuda::GpuMat>& imgs) override {}
            #endif
            /** \阈值图像 **/
            cv::Mat averageImg;
            /** \调制阈值图像 **/
            cv::Mat conditionImg;
            /** \包裹相位图像 **/
            cv::Mat wrapImg;
            /** \参考相位平面 **/
            const cv::Mat& refAbsImg;
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
            void dataInit_Thread_SIMD(const int rows,const int cols,const cv::Point2i region);
            /**
             * @brief 线程入口函数，多线程解相
             *
             * @param rows 输入，行数
             * @param cols 输入，列数
             * @param region 输入，图像操作范围
             * @param absolutePhaseImg 输入输出，绝对相位图片
             */
            void mutiplyThreadUnwrap(const int rows,const int cols,const cv::Point2i region,cv::Mat& absolutePhaseImg);
            /**
             * @brief 计算包裹相位
             * @param lhs 输入，I3-I1
             * @param rhs 输入，I0-I2
             * @param wrapImg 输入输出，包裹相位
             * @param threads 线程数
             */
            void atan2M(const cv::Mat& lhs,const cv::Mat& rhs,cv::Mat& wrapImg,const int threads = 16);
            /**
             * @brief 线程入口函数，多线程计算包裹相位
             * @param lhs 输入，I3-I1
             * @param rhs 输入，I0-I2
             * @param region 输入，多线程区间
             * @param wrapImg 输入输出，包裹相位
             */
            void SIMD_WrapImg(const cv::Mat& lhs,const cv::Mat& rhs,const cv::Point2i& region,cv::Mat& wrapImg);
            /** \全部拍摄图片 0——3：相移 4——9：格雷 10？颜色 **/
            std::vector<cv::Mat> imgs;
            /** \线程数 **/
            const int threads;
    };
}

#endif // !FourStepSixGrayCodeMaster_CPU_H
