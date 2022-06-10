/**
 * @file UnwrapMaster.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  解相器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef PhaseSolver_H
#define PhaseSolver_H
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#ifdef  CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda/std/functional>
#include <cuda/std/cmath>
#include <device_functions.h>
#endif

#include <fstream>
#include <thread>
#include <immintrin.h>
#include <vector>

namespace PhaseSolverType {
    #ifdef CUDA
    namespace cudaFunc {
        //分区间相位展开解调制CUDA主机端调用函数
        void solvePhasePrepare_DevidedSpace(const cv::cuda::GpuMat& shift_0_0_, const cv::cuda::GpuMat & shift_0_1_, const cv::cuda::GpuMat & shift_0_2_, const cv::cuda::GpuMat & gray_0_,
                                            const cv::cuda::GpuMat& shift_1_0_, const cv::cuda::GpuMat & shift_1_1_, const cv::cuda::GpuMat & shift_1_2_, const cv::cuda::GpuMat & gray_1_,
                                            const cv::cuda::GpuMat& shift_2_0_, const cv::cuda::GpuMat & shift_2_1_, const cv::cuda::GpuMat & shift_2_2_, const cv::cuda::GpuMat & gray_2_,
                                            const cv::cuda::GpuMat& shift_3_0_, const cv::cuda::GpuMat & shift_3_1_, const cv::cuda::GpuMat & shift_3_2_, const cv::cuda::GpuMat & gray_3_,
                                            const int rows, const int cols,
                                            cv::cuda::GpuMat & wrapImg1_1_, cv::cuda::GpuMat & wrapImg1_2_, cv::cuda::GpuMat & wrapImg1_3_, cv::cuda::GpuMat & conditionImg_1_,
                                            cv::cuda::GpuMat & wrapImg2_1_, cv::cuda::GpuMat & wrapImg2_2_, cv::cuda::GpuMat & wrapImg2_3_, cv::cuda::GpuMat & conditionImg_2_,
                                            cv::cuda::GpuMat & wrapImg3_1_, cv::cuda::GpuMat & wrapImg3_2_, cv::cuda::GpuMat & wrapImg3_3_, cv::cuda::GpuMat & conditionImg_3_,
                                            cv::cuda::GpuMat & wrapImg4_1_, cv::cuda::GpuMat & wrapImg4_2_, cv::cuda::GpuMat & wrapImg4_3_, cv::cuda::GpuMat & conditionImg_4_,
                                            cv::cuda::GpuMat & floor_K, const dim3 block, cv::cuda::Stream& cvStream);
        //分区间相位展开解相CUDA主机端调用函数
        void solvePhase_DevidedSpace(const cv::cuda::GpuMat & absolutImgWhite, const int rows, const int cols,
                                     const cv::cuda::GpuMat & wrapImg_1_0_, const cv::cuda::GpuMat & wrapImg_1_1_, const cv::cuda::GpuMat & wrapImg_1_2_, const cv::cuda::GpuMat & conditionImg_1_, cv::cuda::GpuMat & unwrapImg_1_,
                                     const cv::cuda::GpuMat & wrapImg_2_0_, const cv::cuda::GpuMat & wrapImg_2_1_, const cv::cuda::GpuMat & wrapImg_2_2_, const cv::cuda::GpuMat & conditionImg_2_, cv::cuda::GpuMat & unwrapImg_2_,
                                     const cv::cuda::GpuMat & wrapImg_3_0_, const cv::cuda::GpuMat & wrapImg_3_1_, const cv::cuda::GpuMat & wrapImg_3_2_, const cv::cuda::GpuMat & conditionImg_3_, cv::cuda::GpuMat & unwrapImg_3_,
                                     const cv::cuda::GpuMat & wrapImg_4_0_, const cv::cuda::GpuMat & wrapImg_4_1_, const cv::cuda::GpuMat & wrapImg_4_2_, const cv::cuda::GpuMat & conditionImg_4_, cv::cuda::GpuMat & unwrapImg_4_, cv::cuda::GpuMat & floor_K, const dim3 block, cv::cuda::Stream& cvStream);
        //位移格雷码解调制CUDA主机端调用函数
        void solvePhasePrepare_ShiftGray(const cv::cuda::GpuMat & shift_0_0_, const cv::cuda::GpuMat & shift_0_1_, const cv::cuda::GpuMat & shift_0_2_,
                                         const cv::cuda::GpuMat & shift_1_0_, const cv::cuda::GpuMat & shift_1_1_, const cv::cuda::GpuMat & shift_1_2_,
                                         const cv::cuda::GpuMat & gray_0_, const cv::cuda::GpuMat & gray_1_, const cv::cuda::GpuMat & gray_2_, const cv::cuda::GpuMat & gray_3_,
                                         const int rows, const int cols,
                                         cv::cuda::GpuMat & wrapImg1, cv::cuda::GpuMat & conditionImg_1_,
                                         cv::cuda::GpuMat & wrapImg2, cv::cuda::GpuMat & conditionImg_2_,
                                         cv::cuda::GpuMat & floor_K, const dim3 block, cv::cuda::Stream& cvStream);
        //位移格雷码解相CUDA主机端调用函数
        void solvePhase_ShiftGray(const cv::cuda::GpuMat & absolutImgWhite,const int rows, const int cols,
                                  const cv::cuda::GpuMat & wrapImg_1_, const cv::cuda::GpuMat & conditionImg_1_, cv::cuda::GpuMat & unwrapImg_1_,
                                  const cv::cuda::GpuMat & wrapImg_2_, const cv::cuda::GpuMat & conditionImg_2_, cv::cuda::GpuMat & unwrapImg_2_,cv::cuda::GpuMat & floor_K, const dim3 block, cv::cuda::Stream& cvStream);
        //四步五位互补格雷码解调制CUDA主机端调用函数
        void solvePhasePrepare_FourStepSixGray(const cv::cuda::GpuMat& shift_0_, const cv::cuda::GpuMat& shift_1_, const cv::cuda::GpuMat& shift_2_, const cv::cuda::GpuMat& shift_3_,
                                               const int rows, const int cols, cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& averageImg, cv::cuda::GpuMat& conditionImg, const dim3 block, cv::cuda::Stream& cvStream);
        //四步五位互补格雷码解相CUDA主机端调用函数
        void solvePhase_FourStepSixGray(const cv::cuda::GpuMat& Gray_0_, const cv::cuda::GpuMat& Gray_1_, const cv::cuda::GpuMat& Gray_2_, const cv::cuda::GpuMat& Gray_3_, const cv::cuda::GpuMat& Gray_4_, const cv::cuda::GpuMat& Gray_5_,
                                        const int rows, const int cols, const cv::cuda::GpuMat& averageImg, const cv::cuda::GpuMat& conditionImg, const cv::cuda::GpuMat& wrapImg, cv::cuda::GpuMat& unwrapImg, const dim3 block, cv::cuda::Stream& cvStream);
    }
    #endif
    class PhaseSolver{
        public:
            /**
             * @brief 默认构造函数
             */
            PhaseSolver();
            /**
             * @brief 析构函数
             */
            virtual ~PhaseSolver();
            /**
             * @brief 解相
             * @param absolutePhaseImg 输入/输出，绝对相位
             */
            virtual void getUnwrapPhaseImg(cv::Mat& absolutePhaseImg) = 0;
            /**
             * @brief 获取纹理图片
             * @param textureImg 输入/输出，纹理图片
             */
            virtual void getTextureImg(cv::Mat& textureImg) = 0;
            #ifdef CUDA
            /**
             * @brief 上传图片
             * @param imgs 输入，原始图片（CV_8UC1)
             * @param stream 输入，异步流
             */
            virtual void changeSourceImg(std::vector<cv::Mat>& imgs, cv::cuda::Stream& stream) = 0;
            /**
             * @brief 上传图片
             * @param imgs 输入，设备端图片（CV_8UC1)
             */
            virtual void changeSourceImg(std::vector<cv::cuda::GpuMat>& imgs) = 0;
            /**
             * @brief 解相
             * @param absolutePhaseImg 输入/输出，绝对相位
             * @param stream 输入，异步流
             */
            virtual void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& absolutePhaseImg, cv::cuda::Stream& stream) = 0;
            /**
             * @brief 获取包裹相位
             * @param wrapImg 输入，包裹相位
             * @param stream 输入，异步流
             */
            //virtual void getWrapPhaseImg(cv::cuda::GpuMat& wrapImg, cv::cuda::Stream& stream) = 0;
            /**
             * @brief 获取纹理图片
             * @param textureImg 输入/输出，纹理图片
             */
            virtual void getTextureImg(std::vector<cv::cuda::GpuMat>& textureImg) = 0;
            #endif
            /**
             * @brief 上传图片
             * @param imgs 输入，原始图片
             */
            virtual void changeSourceImg(std::vector<cv::Mat>& imgs) = 0;
            /**
             * @brief 获取包裹相位
             * @param wrapImg 输入，包裹相位
             */
            virtual void getWrapPhaseImg(cv::Mat& wrapImg,cv::Mat& conditionImg) = 0;
        protected:

        private:
    };
}
#endif // !PhaseSolver_H
