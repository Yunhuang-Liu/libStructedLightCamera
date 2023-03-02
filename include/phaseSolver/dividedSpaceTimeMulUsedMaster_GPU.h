/**
 * @file dividedSpaceTimeMulUsedMaster_GPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  分区间相位展开+时间复用GPU加速解相器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef PHASESOLVER_DIVIDEDTIMEMULUSEDMASTER_GPU_H_
#define PHASESOLVER_DIVIDEDTIMEMULUSEDMASTER_GPU_H_

#include "phaseSolver.h"

/** @brief 结构光库 */
namespace sl {
    /** @brief 解相库 */
    namespace phaseSolver {
        /**
          * @brief 分区间相位展开+时间复用格雷码GPU解相器
          */
        class DividedSpaceTimeMulUsedMaster_GPU : public PhaseSolver {
        public:
            /**
             * @brief 构造函数
             * @param refImgWhite_ 输入，参考绝对相位
             * @param block 输入，Block尺寸
             */
            DividedSpaceTimeMulUsedMaster_GPU(const cv::Mat &refImgWhite_,
                                              const dim3 block = dim3(32, 8));
            /**
             * @brief 构造函数
             * @param imgs 输入，原始图片
             * @param refImgWhite 输入，参考绝对相位
             * @param block 输入，Block尺寸
             */
            DividedSpaceTimeMulUsedMaster_GPU(std::vector<cv::Mat> &imgs,
                                              const cv::Mat &refImgWhite,
                                              const dim3 block = dim3(32, 8));
            /**
             * @brief 析构函数
             */
            ~DividedSpaceTimeMulUsedMaster_GPU();
            /**
             * @brief 解相
             * @param absolutePhaseImg 输入/输出，绝对相位图
             * @param pStream 输入，异步流
             */
            void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat> &unwrapImg,
                                   cv::cuda::Stream &pStream) override;
            /**
             * @brief 上传原始图片
             * @param imgs 输入，原始图片
             * @param stream 输入，异步流
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
             * @brief 获取包裹相位
             * @param stream 输入，异步流
             */
            void getWrapPhaseImg(cv::cuda::Stream &pStream);
            /**
             * @brief 更改源图片,同步版本，暂时未被移除
             * @param imgs 输入，原始图片
             */
            void changeSourceImg(std::vector<cv::Mat> &imgs) override;

        private:
            /** \block尺寸 **/
            const dim3 block;
            void getUnwrapPhaseImg(cv::Mat &) override {}
            void getWrapPhaseImg(cv::Mat &, cv::Mat &) override {}
            void getTextureImg(cv::Mat &textureImg) override {}
            /** \第一帧阈值图像 **/
            cv::cuda::GpuMat averageImg_1_device;
            /** \第二帧阈值图像 **/
            cv::cuda::GpuMat averageImg_2_device;
            /** \第三帧阈值图像 **/
            cv::cuda::GpuMat averageImg_3_device;
            /** \第四帧阈值图像 **/
            cv::cuda::GpuMat averageImg_4_device;
            /** \第一帧调制图像 **/
            cv::cuda::GpuMat conditionImg_1_device;
            /** \第二帧调制图像 **/
            cv::cuda::GpuMat conditionImg_2_device;
            /** \第三帧调制图像 **/
            cv::cuda::GpuMat conditionImg_3_device;
            /** \第四帧调制图像 **/
            cv::cuda::GpuMat conditionImg_4_device;
            /** \第一帧包裹相位图像—1 **/
            cv::cuda::GpuMat wrapImg1_1_device;
            /** \第一帧包裹相位图像—2 **/
            cv::cuda::GpuMat wrapImg1_2_device;
            /** \第一帧包裹相位图像—3 **/
            cv::cuda::GpuMat wrapImg1_3_device;
            /** \第二帧包裹相位图像—1 **/
            cv::cuda::GpuMat wrapImg2_1_device;
            /** \第二帧包裹相位图像—2 **/
            cv::cuda::GpuMat wrapImg2_2_device;
            /** \第二帧包裹相位图像—3 **/
            cv::cuda::GpuMat wrapImg2_3_device;
            /** \第三帧包裹相位图像—1 **/
            cv::cuda::GpuMat wrapImg3_1_device;
            /** \第三帧包裹相位图像—2 **/
            cv::cuda::GpuMat wrapImg3_2_device;
            /** \第三帧包裹相位图像—3 **/
            cv::cuda::GpuMat wrapImg3_3_device;
            /** \第四帧包裹相位图像—1 **/
            cv::cuda::GpuMat wrapImg4_1_device;
            /** \第四帧包裹相位图像—2 **/
            cv::cuda::GpuMat wrapImg4_2_device;
            /** \第四帧包裹相位图像—3 **/
            cv::cuda::GpuMat wrapImg4_3_device;
            /** \参考平面绝对相位 **/
            const cv::cuda::GpuMat refImgWhite_device;
            /** \第一帧图像:GrayCode **/
            cv::cuda::GpuMat img1_1_device;
            /** \第一帧图像:Phase **/
            cv::cuda::GpuMat img1_2_device;
            /** \第一帧图像:Phase **/
            cv::cuda::GpuMat img1_3_device;
            /** \第一帧图像:Phase**/
            cv::cuda::GpuMat img1_4_device;
            /** \第二帧图像:GrayCode **/
            cv::cuda::GpuMat img2_1_device;
            /** \第二帧图像:Phase **/
            cv::cuda::GpuMat img2_2_device;
            /** \第二帧图像:Phase **/
            cv::cuda::GpuMat img2_3_device;
            /** \第二帧图像:Phase **/
            cv::cuda::GpuMat img2_4_device;
            /** \第三帧图像:GrayCode **/
            cv::cuda::GpuMat img3_1_device;
            /** \第三帧图像:Phase **/
            cv::cuda::GpuMat img3_2_device;
            /** \第三帧图像:Phase **/
            cv::cuda::GpuMat img3_3_device;
            /** \第三帧图像:Phase **/
            cv::cuda::GpuMat img3_4_device;
            /** \第四帧图像:GrayCode **/
            cv::cuda::GpuMat img4_1_device;
            /** \第四帧图像:Phase **/
            cv::cuda::GpuMat img4_2_device;
            /** \第四帧图像:Phase **/
            cv::cuda::GpuMat img4_3_device;
            /** \第四帧图像:Phase **/
            cv::cuda::GpuMat img4_4_device;
            /** \第一帧绝对相位 **/
            cv::cuda::GpuMat unwrapImg_1_device;
            /** \第二帧绝对相位 **/
            cv::cuda::GpuMat unwrapImg_2_device;
            /** \第三帧绝对相位 **/
            cv::cuda::GpuMat unwrapImg_3_device;
            /** \第四帧绝对相位 **/
            cv::cuda::GpuMat unwrapImg_4_device;
            /** \解相阶级 **/
            cv::cuda::GpuMat floor_K_device;
            /** \行数 **/
            int rows;
            /** \列数 **/
            int cols;
        };
    }// namespace phaseSolver
}// namespace sl
#endif // PHASESOLVER_DIVIDEDTIMEMULUSEDMASTER_GPU_H_
