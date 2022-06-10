/**
 * @file DividedSpaceTimeMulUsedMaster_GPU.cuh
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  位移格雷码GPU加速解相器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef ShiftGrayCodeUnwrapMaster_GPU_H
#define ShiftGrayCodeUnwrapMaster_GPU_H

#include "./PhaseSolver.h"

namespace PhaseSolverType {
    class ShiftGrayCodeUnwrapMaster_GPU : public PhaseSolver {
    public:
        /**
         * @brief 构造函数
         * @param refImgWhite_ 输入，参考绝对相位
         * @param block 输入，Block尺寸
         */
        ShiftGrayCodeUnwrapMaster_GPU(const dim3 block = dim3(32, 8), const cv::Mat& refImgWhite_ = cv::Mat(1, 1, CV_32FC1));
        /**
         * @brief 构造函数
         * @param imgs 输入，原始图片
         * @param refImgWhite 输入，参考绝对相位
         * @param block 输入，Block尺寸
         */
        ShiftGrayCodeUnwrapMaster_GPU(std::vector<cv::Mat>& imgs,const cv::Mat& refImgWhite = cv::Mat(1,1,CV_32FC1), const dim3 block = dim3(32, 8));
        /**
         * @brief 析构函数
         */
        ~ShiftGrayCodeUnwrapMaster_GPU();
        /**
         * @brief 解相
         * @param unwrapImg 输入/输出，绝对相位图
         * @param pStream 输入，异步流
         */
        void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& pStream);
        /**
         * @brief 上传图片
         * @param imgs 输入，原始图片
         * @param stream 输入，异步流
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs, cv::cuda::Stream& stream) override;
        /**
         * @brief 上传原始图片
         * @param imgs 输入，设备端图片（CV_8UC1)
         */
        void changeSourceImg(std::vector<cv::cuda::GpuMat>& imgs) override;
        /**
         * @brief 获取纹理图片
         * @param textureImg 输入/输出，纹理图片
         */
        void getTextureImg(std::vector<cv::cuda::GpuMat>& textureImg) override;
    protected:
        /**
         * @brief 上传图片
         * @param imgs 输入，原始图片
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs) override;
        /**
         * @brief 获取包裹相位
         * @param pStream 输入，异步流
         */
        void getWrapPhaseImg(cv::cuda::Stream& pStream);
    private:
        void getUnwrapPhaseImg(cv::Mat&) override {}
        void getWrapPhaseImg(cv::Mat&, cv::Mat&) override {}
        void getTextureImg(cv::Mat& textureImg) override {}
        /** \参考平面绝对相位 **/
        const cv::cuda::GpuMat refImgWhite_device;
        /** \第一帧阈值图像 **/
        cv::cuda::GpuMat averageImg_device;
        /** \第一帧调制图像 **/
        cv::cuda::GpuMat conditionImg_1_device;
        /** \第二帧调制图像 **/
        cv::cuda::GpuMat conditionImg_2_device;
        /** \第一帧包裹相位图像—1 **/
        cv::cuda::GpuMat wrapImg1_device;
        /** \第二帧包裹相位图像—1 **/
        cv::cuda::GpuMat wrapImg2_device;
        /** \第一帧图像:Phase **/
        cv::cuda::GpuMat img1_1_device;
        /** \第一帧图像:Phase **/
        cv::cuda::GpuMat img1_2_device;
        /** \第一帧图像:Phase **/
        cv::cuda::GpuMat img1_3_device;
        /** \第二帧图像:Phase **/
        cv::cuda::GpuMat img2_1_device;
        /** \第二帧图像:Phase **/
        cv::cuda::GpuMat img2_2_device;
        /** \第二帧图像:Phase **/
        cv::cuda::GpuMat img2_3_device;
        /** \GrayCode 1**/
        cv::cuda::GpuMat imgG_1_device;
        /** \GrayCode 2 **/
        cv::cuda::GpuMat imgG_2_device;
        /** \GrayCode 3 **/
        cv::cuda::GpuMat imgG_3_device;
        /** \GrayCode 4 **/
        cv::cuda::GpuMat imgG_4_device;
        /** \第一帧绝对相位 **/
        cv::cuda::GpuMat unwrapImg_1_device;
        /** \第二帧绝对相位 **/
        cv::cuda::GpuMat unwrapImg_2_device;
        /** \解相阶级 **/
        cv::cuda::GpuMat floor_K_device;
        /** \block尺寸 **/
        const dim3 block;
        /** \行数 **/
        int rows;
        /** \列数 **/
        int cols;
    };
}

#endif // !ShiftGrayCodeUnwrapMaster_GPU_H
