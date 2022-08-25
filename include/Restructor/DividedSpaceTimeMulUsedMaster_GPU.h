/**
 * @file DividedSpaceTimeMulUsedMaster_GPU.cuh
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  ��������λչ��+ʱ�临��GPU���ٽ�����
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef DividedSpaceTimeMulUsedMaster_GPU_H
#define DividedSpaceTimeMulUsedMaster_GPU_H

#include "./PhaseSolver.h"

namespace PhaseSolverType {
    /**
     * @brief ��������λչ��+ʱ�临�ø�����GPU������
     */
    class DividedSpaceTimeMulUsedMaster_GPU : public PhaseSolver {
    public:
        /**
         * @brief ���캯��
         * @param refImgWhite_ ���룬�ο�������λ
         * @param block ���룬Block�ߴ�
         */
        DividedSpaceTimeMulUsedMaster_GPU(const cv::Mat& refImgWhite_, const dim3 block = dim3(32, 8));
        /**
         * @brief ���캯��
         * @param imgs ���룬ԭʼͼƬ
         * @param refImgWhite ���룬�ο�������λ
         * @param block ���룬Block�ߴ�
         */
        DividedSpaceTimeMulUsedMaster_GPU(std::vector<cv::Mat> &imgs,const cv::Mat& refImgWhite, const dim3 block = dim3(32, 8));
        /**
         * @brief ��������
         */
        ~DividedSpaceTimeMulUsedMaster_GPU();
        /**
         * @brief ����
         * @param absolutePhaseImg ����/�����������λͼ
         * @param pStream ���룬�첽��
         */
        void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& pStream) override ;
        /**
         * @brief �ϴ�ԭʼͼƬ
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
         * @brief ��ȡ����ͼƬ���Ҷȸ����ͣ�
         * @param textureImg ����/���������ͼƬ
         */
        void getTextureImg(std::vector<cv::cuda::GpuMat>& textureImg) override;
    protected:
        /**
         * @brief ��ȡ������λ
         * @param stream ���룬�첽��
         */
        void getWrapPhaseImg(cv::cuda::Stream& pStream);
        /**
         * @brief ����ԴͼƬ,ͬ���汾����ʱδ���Ƴ�
         * @param imgs ���룬ԭʼͼƬ
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs) override;
    private:
        void getUnwrapPhaseImg(cv::Mat&) override {}
        void getWrapPhaseImg(cv::Mat&, cv::Mat&) override {}
        void getTextureImg(cv::Mat& textureImg) override {}
        /** \��һ֡��ֵͼ�� **/
        cv::cuda::GpuMat averageImg_1_device;
        /** \�ڶ�֡��ֵͼ�� **/
        cv::cuda::GpuMat averageImg_2_device;
        /** \����֡��ֵͼ�� **/
        cv::cuda::GpuMat averageImg_3_device;
        /** \����֡��ֵͼ�� **/
        cv::cuda::GpuMat averageImg_4_device;
        /** \��һ֡����ͼ�� **/
        cv::cuda::GpuMat conditionImg_1_device;
        /** \�ڶ�֡����ͼ�� **/
        cv::cuda::GpuMat conditionImg_2_device;
        /** \����֡����ͼ�� **/
        cv::cuda::GpuMat conditionImg_3_device;
        /** \����֡����ͼ�� **/
        cv::cuda::GpuMat conditionImg_4_device;
        /** \��һ֡������λͼ��1 **/
        cv::cuda::GpuMat wrapImg1_1_device;
        /** \��һ֡������λͼ��2 **/
        cv::cuda::GpuMat wrapImg1_2_device;
        /** \��һ֡������λͼ��3 **/
        cv::cuda::GpuMat wrapImg1_3_device;
        /** \�ڶ�֡������λͼ��1 **/
        cv::cuda::GpuMat wrapImg2_1_device;
        /** \�ڶ�֡������λͼ��2 **/
        cv::cuda::GpuMat wrapImg2_2_device;
        /** \�ڶ�֡������λͼ��3 **/
        cv::cuda::GpuMat wrapImg2_3_device;
        /** \����֡������λͼ��1 **/
        cv::cuda::GpuMat wrapImg3_1_device;
        /** \����֡������λͼ��2 **/
        cv::cuda::GpuMat wrapImg3_2_device;
        /** \����֡������λͼ��3 **/
        cv::cuda::GpuMat wrapImg3_3_device;
        /** \����֡������λͼ��1 **/
        cv::cuda::GpuMat wrapImg4_1_device;
        /** \����֡������λͼ��2 **/
        cv::cuda::GpuMat wrapImg4_2_device;
        /** \����֡������λͼ��3 **/
        cv::cuda::GpuMat wrapImg4_3_device;
        /** \�ο�ƽ�������λ **/
        const cv::cuda::GpuMat refImgWhite_device;
        /** \��һ֡ͼ��:GrayCode **/
        cv::cuda::GpuMat img1_1_device;
        /** \��һ֡ͼ��:Phase **/
        cv::cuda::GpuMat img1_2_device;
        /** \��һ֡ͼ��:Phase **/
        cv::cuda::GpuMat img1_3_device;
        /** \��һ֡ͼ��:Phase**/
        cv::cuda::GpuMat img1_4_device;
        /** \�ڶ�֡ͼ��:GrayCode **/
        cv::cuda::GpuMat img2_1_device;
        /** \�ڶ�֡ͼ��:Phase **/
        cv::cuda::GpuMat img2_2_device;
        /** \�ڶ�֡ͼ��:Phase **/
        cv::cuda::GpuMat img2_3_device;
        /** \�ڶ�֡ͼ��:Phase **/
        cv::cuda::GpuMat img2_4_device;
        /** \����֡ͼ��:GrayCode **/
        cv::cuda::GpuMat img3_1_device;
        /** \����֡ͼ��:Phase **/
        cv::cuda::GpuMat img3_2_device;
        /** \����֡ͼ��:Phase **/
        cv::cuda::GpuMat img3_3_device;
        /** \����֡ͼ��:Phase **/
        cv::cuda::GpuMat img3_4_device;
        /** \����֡ͼ��:GrayCode **/
        cv::cuda::GpuMat img4_1_device;
        /** \����֡ͼ��:Phase **/
        cv::cuda::GpuMat img4_2_device;
        /** \����֡ͼ��:Phase **/
        cv::cuda::GpuMat img4_3_device;
        /** \����֡ͼ��:Phase **/
        cv::cuda::GpuMat img4_4_device;
        /** \��һ֡������λ **/
        cv::cuda::GpuMat unwrapImg_1_device;
        /** \�ڶ�֡������λ **/
        cv::cuda::GpuMat unwrapImg_2_device;
        /** \����֡������λ **/
        cv::cuda::GpuMat unwrapImg_3_device;
        /** \����֡������λ **/
        cv::cuda::GpuMat unwrapImg_4_device;
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


#endif // !DividedSpaceTimeMulUsedMaster_GPU_H
