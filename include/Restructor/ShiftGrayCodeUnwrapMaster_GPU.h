/**
 * @file DividedSpaceTimeMulUsedMaster_GPU.cuh
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  λ�Ƹ�����GPU���ٽ�����
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
         * @brief ���캯��
         * @param refImgWhite_ ���룬�ο�������λ
         * @param block ���룬Block�ߴ�
         */
        ShiftGrayCodeUnwrapMaster_GPU(const dim3 block = dim3(32, 8), const cv::Mat& refImgWhite_ = cv::Mat(1, 1, CV_32FC1));
        /**
         * @brief ���캯��
         * @param imgs ���룬ԭʼͼƬ
         * @param refImgWhite ���룬�ο�������λ
         * @param block ���룬Block�ߴ�
         */
        ShiftGrayCodeUnwrapMaster_GPU(std::vector<cv::Mat>& imgs,const cv::Mat& refImgWhite = cv::Mat(1,1,CV_32FC1), const dim3 block = dim3(32, 8));
        /**
         * @brief ��������
         */
        ~ShiftGrayCodeUnwrapMaster_GPU();
        /**
         * @brief ����
         * @param unwrapImg ����/�����������λͼ
         * @param pStream ���룬�첽��
         */
        void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& pStream);
        /**
         * @brief �ϴ�ͼƬ
         * @param imgs ���룬ԭʼͼƬ
         * @param stream ���룬�첽��
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs, cv::cuda::Stream& stream) override;
        /**
         * @brief �ϴ�ԭʼͼƬ
         * @param imgs ���룬�豸��ͼƬ��CV_8UC1)
         */
        void changeSourceImg(std::vector<cv::cuda::GpuMat>& imgs) override;
        /**
         * @brief ��ȡ����ͼƬ
         * @param textureImg ����/���������ͼƬ
         */
        void getTextureImg(std::vector<cv::cuda::GpuMat>& textureImg) override;
    protected:
        /**
         * @brief �ϴ�ͼƬ
         * @param imgs ���룬ԭʼͼƬ
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs) override;
        /**
         * @brief ��ȡ������λ
         * @param pStream ���룬�첽��
         */
        void getWrapPhaseImg(cv::cuda::Stream& pStream);
    private:
        void getUnwrapPhaseImg(cv::Mat&) override {}
        void getWrapPhaseImg(cv::Mat&, cv::Mat&) override {}
        void getTextureImg(cv::Mat& textureImg) override {}
        /** \�ο�ƽ�������λ **/
        const cv::cuda::GpuMat refImgWhite_device;
        /** \��һ֡��ֵͼ�� **/
        cv::cuda::GpuMat averageImg_device;
        /** \��һ֡����ͼ�� **/
        cv::cuda::GpuMat conditionImg_1_device;
        /** \�ڶ�֡����ͼ�� **/
        cv::cuda::GpuMat conditionImg_2_device;
        /** \��һ֡������λͼ��1 **/
        cv::cuda::GpuMat wrapImg1_device;
        /** \�ڶ�֡������λͼ��1 **/
        cv::cuda::GpuMat wrapImg2_device;
        /** \��һ֡ͼ��:Phase **/
        cv::cuda::GpuMat img1_1_device;
        /** \��һ֡ͼ��:Phase **/
        cv::cuda::GpuMat img1_2_device;
        /** \��һ֡ͼ��:Phase **/
        cv::cuda::GpuMat img1_3_device;
        /** \�ڶ�֡ͼ��:Phase **/
        cv::cuda::GpuMat img2_1_device;
        /** \�ڶ�֡ͼ��:Phase **/
        cv::cuda::GpuMat img2_2_device;
        /** \�ڶ�֡ͼ��:Phase **/
        cv::cuda::GpuMat img2_3_device;
        /** \GrayCode 1**/
        cv::cuda::GpuMat imgG_1_device;
        /** \GrayCode 2 **/
        cv::cuda::GpuMat imgG_2_device;
        /** \GrayCode 3 **/
        cv::cuda::GpuMat imgG_3_device;
        /** \GrayCode 4 **/
        cv::cuda::GpuMat imgG_4_device;
        /** \��һ֡������λ **/
        cv::cuda::GpuMat unwrapImg_1_device;
        /** \�ڶ�֡������λ **/
        cv::cuda::GpuMat unwrapImg_2_device;
        /** \����׼� **/
        cv::cuda::GpuMat floor_K_device;
        /** \block�ߴ� **/
        const dim3 block;
        /** \���� **/
        int rows;
        /** \���� **/
        int cols;
    };
}

#endif // !ShiftGrayCodeUnwrapMaster_GPU_H
