/**
 * @file FourStepSixGrayCodeMaster_GPU.cuh
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU加速解相器(四步五位互补格雷码)
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef FourStepSixGrayCodeMaster_GPU_H
#define FourStepSixGrayCodeMaster_GPU_H
#include "./PhaseSolver.h"

namespace PhaseSolverType {
    /**
     * @brief 互补格雷码四步相移解码器(4+6 | CUDA)
     *
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
        FourStepSixGrayCodeMaster_GPU(std::vector<cv::Mat>& imgs, const dim3 block_ = dim3(32, 8));
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
        void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& pStream) override;
        /**
         * @brief 上传图片
         * @param imgs 输入，原始图片
         * @param pStream 输入，异步流
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs, cv::cuda::Stream& stream) override;
    protected:
        /**
         * @brief 上传图片
         * @param imgs 输入，原始图片
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs) override;
        /**
         * @brief 获取包裹相位
         */
        void getWrapPhaseImg();
    private:
        /**
         * @brief 解相
         * @param absolutePhaseImg
         */
        void getUnwrapPhaseImg(cv::Mat&) override{}
        /** \GPU图像 
         *  \全部拍摄图片 0——3：相移 4——9：格雷 10？颜色
         **/
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
}
#endif // !FourStepSixGrayCodeMaster_GPU_H
