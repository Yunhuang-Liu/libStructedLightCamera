/**
 * @file fourFloorFouStepMaster_GPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU加速解相器(四步四灰度时间复用格雷码)
 * @version 0.1
 * @date 2022-8-8
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef PHASESOLVER_FOURFLOORSTEPMASTER_GPU_H_
#define PHASESOLVER_FOURFLOORSTEPMASTER_GPU_H_

#include <phaseSolver/PhaseSolver.h>

/** @brief 结构光库 */
namespace sl {
    /** @brief 解相库 */
    namespace phaseSolver {
        /**
         * @brief 四步四灰度时间复用格雷码解相器(1+2 - 1+2 --- | CUDA)
         */
        class FourFloorFourStepMaster_GPU : public PhaseSolver {
        public:
            /**
             * @brief 构造函数
             * @param block_ 输入，Block尺寸
             */
            FourFloorFourStepMaster_GPU(const dim3 block_ = dim3(32, 8));
            /**
             * @brief 构造函数
             * @param imgs 输入，原始图片
             * @param block_ 输入，Block尺寸
             */
            FourFloorFourStepMaster_GPU(std::vector<cv::Mat> &imgs,
                                        const dim3 block_ = dim3(32, 8));
            /**
             * @brief 析构函数
             */
            ~FourFloorFourStepMaster_GPU();
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
            /**
             * @brief 获取包裹相位
             */
            void getWrapPhaseImg();

        private:
            /** \block尺寸 **/
            const dim3 block;
            void getUnwrapPhaseImg(cv::Mat &) override {}
            void getWrapPhaseImg(cv::Mat &, cv::Mat &) override {}
            void getTextureImg(cv::Mat &textureImg) override {}
            /** \GPU图像 全部拍摄图片 0——3：相移 4——9：格雷 **/
            std::vector<cv::cuda::GpuMat> imgs_device;
            /** \阶数图像 **/
            cv::cuda::GpuMat floorImg_device;
            /** \阶数图像- 0 - **/
            cv::cuda::GpuMat floorImg_0_device;
            /** \阶数图像- 1 - **/
            cv::cuda::GpuMat floorImg_1_device;
            /** \中值滤波图像- 0 - **/
            cv::cuda::GpuMat medianFilter_0_;
            /** \中值滤波图像- 1 - **/
            cv::cuda::GpuMat medianFilter_1_;
            /** \调制图像 **/
            cv::cuda::GpuMat conditionImg_device;
            /** \包裹相位图像 **/
            cv::cuda::GpuMat wrapImg_device;
            /** \Kmeans阈值 **/
            cv::cuda::GpuMat threshodVal;
            /** \Kmeans群之和 **/
            cv::cuda::GpuMat threshodAdd;
            /** \Kmeans群数量 **/
            cv::cuda::GpuMat count;
            /** \包裹相位图像 **/
            cv::cuda::GpuMat conditionImgCopy;
            /** \行数 **/
            int rows;
            /** \列数 **/
            int cols;
            /** \当前连续帧数 **/
            int currentFrame;
        };
    }// namespace phaseSolver
}// namespace sl
#endif// PHASESOLVER_FOURFLOORSTEPMASTER_GPU_H_