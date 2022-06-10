/**
 * @file FourStepSixGrayCodeMaster_GPU.cuh
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU���ٽ�����(�Ĳ���λ����������)
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
     * @brief �����������Ĳ����ƽ�����(4+6 | CUDA)
     *
     */
    class FourStepSixGrayCodeMaster_GPU : public PhaseSolver {
    public:
        /**
         * @brief ���캯��
         * @param block_ ���룬Block�ߴ�
         */
        FourStepSixGrayCodeMaster_GPU(const dim3 block_ = dim3(32, 8));
        /**
         * @brief ���캯��
         * @param imgs ���룬ԭʼͼƬ
         * @param block_ ���룬Block�ߴ�
         */
        FourStepSixGrayCodeMaster_GPU(std::vector<cv::Mat>& imgs, const dim3 block_ = dim3(32, 8));
        /**
         * @brief ��������
         */
        ~FourStepSixGrayCodeMaster_GPU();
        /**
         * @brief ����
         *
         * @param unwrapImg ����/�����������λͼ
         * @param pStream ���룬�첽��
         */
        void getUnwrapPhaseImg(std::vector<cv::cuda::GpuMat>& unwrapImg, cv::cuda::Stream& pStream) override;
        /**
         * @brief �ϴ�ͼƬ
         * @param imgs ���룬ԭʼͼƬ
         * @param pStream ���룬�첽��
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
         * @brief �ϴ�ͼƬ
         * @param imgs ���룬ԭʼͼƬ
         */
        void changeSourceImg(std::vector<cv::Mat>& imgs) override;
        /**
         * @brief ��ȡ������λ
         */
        void getWrapPhaseImg();
    private:
        void getUnwrapPhaseImg(cv::Mat&) override{}
        void getWrapPhaseImg(cv::Mat&, cv::Mat&) override {}
        void getTextureImg(cv::Mat& textureImg) override {}
        /** \GPUͼ�� 
         *  \ȫ������ͼƬ 0����3������ 4����9������ 10����ɫ
         **/
        std::vector<cv::cuda::GpuMat> imgs_device;
        /** \��ֵͼ�� **/
        cv::cuda::GpuMat averageImg_device;
        /** \����ͼ�� **/
        cv::cuda::GpuMat conditionImg_device;
        /** \������λͼ�� **/
        cv::cuda::GpuMat wrapImg_device;
        /** \block�ߴ� **/
        const dim3 block;
        /** \���� **/
        int rows;
        /** \���� **/
        int cols;
    };
}
#endif // !FourStepSixGrayCodeMaster_GPU_H
