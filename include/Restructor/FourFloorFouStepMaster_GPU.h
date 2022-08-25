/**
 * @file FourFloorFouStepMaster_GPU.cuh
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU���ٽ�����(�Ĳ��ĻҶ�ʱ�临�ø�����)
 * @version 0.1
 * @date 2022-8-8
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef FourFloorFourStepMaster_GPU_H
#define FourFloorFourStepMaster_GPU_H
#include "./PhaseSolver.h"

namespace PhaseSolverType {
    /**
     * @brief �Ĳ��ĻҶ�ʱ�临�ø����������(1+2 - 1+2 --- | CUDA)
     *
     */
    class FourFloorFourStepMaster_GPU : public PhaseSolver {
    public:
        /**
         * @brief ���캯��
         * @param block_ ���룬Block�ߴ�
         */
        FourFloorFourStepMaster_GPU(const dim3 block_ = dim3(32, 8));
        /**
         * @brief ���캯��
         * @param imgs ���룬ԭʼͼƬ
         * @param block_ ���룬Block�ߴ�
         */
        FourFloorFourStepMaster_GPU(std::vector<cv::Mat>& imgs, const dim3 block_ = dim3(32, 8));
        /**
         * @brief ��������
         */
        ~FourFloorFourStepMaster_GPU();
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
        void getUnwrapPhaseImg(cv::Mat&) override {}
        void getWrapPhaseImg(cv::Mat&, cv::Mat&) override {}
        void getTextureImg(cv::Mat& textureImg) override {}
        /** \GPUͼ��
         *  \ȫ������ͼƬ 0����3������ 4����9������ 10����ɫ
         **/
        std::vector<cv::cuda::GpuMat> imgs_device;
        /** \����ͼ�� **/
        cv::cuda::GpuMat floorImg_device;
        /** \����ͼ��- 0 - **/
        cv::cuda::GpuMat floorImg_0_device;
        /** \����ͼ��- 1 - **/
        cv::cuda::GpuMat floorImg_1_device; 
        /** \��ֵ�˲�ͼ��- 0 - **/
        cv::cuda::GpuMat medianFilter_0_;
        /** \��ֵ�˲�ͼ��- 1 - **/
        cv::cuda::GpuMat medianFilter_1_;
        /** \����ͼ�� **/
        cv::cuda::GpuMat conditionImg_device;
        /** \������λͼ�� **/
        cv::cuda::GpuMat wrapImg_device;
        /** \Kmeans��ֵ **/
        cv::cuda::GpuMat threshodVal;
        /** \KmeansȺ֮�� **/
        cv::cuda::GpuMat threshodAdd;
        /** \KmeansȺ���� **/
        cv::cuda::GpuMat count;
        /** \������λͼ�� **/
        cv::cuda::GpuMat conditionImgCopy;
        /** \block�ߴ� **/
        const dim3 block;
        /** \���� **/
        int rows;
        /** \���� **/
        int cols;
        /** \��ǰ����֡�� **/
        int currentFrame;
    };
}
#endif // !FourFloorFourStepMaster_GPU_H