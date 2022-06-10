/**
 * @file Restruction_GPU_Divided_Time.cuh
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU�ؽ���
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef Restruction_GPU_H
#define Restruction_GPU_H
#include "Restructor.h"
#include "MatrixsInfo.h"
#include <limits>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

namespace RestructorType {
    class Restructor_GPU : public Restructor {
    public:
        /**
         * @brief ���캯��
         * @param calibrationInfo ���룬�궨��Ϣ
         * @param minDisparity ���룬��С�Ӳ�
         * @param maxDisparity ���룬����Ӳ�
         * @param minDepth ���룬��С���ֵ
         * @param maxDepth ���룬������ֵ
         * @param block ���룬Block�ߴ�
         */
        Restructor_GPU(const Info& calibrationInfo, const int minDisparity = -500, const int maxDisparity = 500,
                       const float minDepth = 170, const float maxDepth = 220, const dim3 block = dim3(32, 8));
        /**
         * @brief ��������
         */
        ~Restructor_GPU();
        /**
         * @brief �ؽ�
         * @param leftAbsImg ���룬�������λ
         * @param rightAbsImg ���룬�Ҿ�����λ
         * @param sysIndex ���룬ͼƬ����
         * @param stream ���룬�첽��
         * @param colorImg ���룬��ɫ����
         */
        void restruction(const cv::cuda::GpuMat& leftAbsImg, const cv::cuda::GpuMat& rightAbsImg,
            const int sysIndex, cv::cuda::Stream& stream, const cv::Mat& colorImg = cv::Mat(1, 1, CV_8UC3)) override;
        /**
         * @brief ��ȡ�������
         * @param index ���룬ͼƬ����
         * @param depthImg ����/��������ͼ
         * @param colorImg ����/���������ͼ
         */
        void download(const int index, cv::cuda::GpuMat& depthImg, cv::cuda::GpuMat& colorImg = cv::cuda::GpuMat(1, 1, CV_8UC3)) override;
    protected:
        /**
         * @brief ӳ���������
         * @param leftImg ���룬�������λ
         * @param rightImg ���룬�Ҿ�����λ
         * @param colorImg ���룬ԭʼ��ɫ����
         * @param depthImg ����/��������ͼ
         * @param mapColorImg ����/���������ͼ
         * @param pStream ���룬�첽��
         */
        void getDepthColorMap(const cv::cuda::GpuMat& leftImg, const cv::cuda::GpuMat& rightImg, const cv::cuda::GpuMat& colorImg, cv::cuda::GpuMat& depthImg, cv::cuda::GpuMat& mapColorImg, cv::cuda::Stream& pStream);
        /**
         * @brief ӳ���������
         * @param leftImg ���룬�������λ
         * @param rightImg ���룬�Ҿ�����λ
         * @param depthImg ����/��������ͼ
         * @param pStream ���룬�첽��
         */
        void getDepthMap(const cv::cuda::GpuMat& leftImg, const cv::cuda::GpuMat& rightImg, cv::cuda::GpuMat& depthImg, cv::cuda::Stream& pStream);
    private:
        //CPU�˺���
        void restruction(const cv::Mat& leftAbsImg, const cv::Mat& rightAbsImg, cv::Mat& depthImgOut, const cv::Mat& colorImg = cv::Mat(1, 1, CV_8UC3), cv::Mat& colorImgOut = cv::Mat(1, 1, CV_8UC3)) override{}
        /** \���ͼ **/
        std::vector<cv::cuda::GpuMat> depthImg_device;
        /** \����ӳ��ͼ **/
        std::vector<cv::cuda::GpuMat> colorImg_device;
        /** \�궨��Ϣ **/
        const Info& calibrationInfo;
        /** \���ӳ����� **/
        Eigen::Matrix4f Q;
        /** \���ӳ����� **/
        Eigen::Matrix3f R1_inv;
        /** \�Ҷ��������ɫ�����ת���� **/
        Eigen::Matrix3f R;
        /** \�Ҷ��������ɫ���λ�ƾ��� **/
        Eigen::Vector3f T;
        /** \��ɫ����ڲξ��� **/
        Eigen::Matrix3f M3;
        /** \��С�Ӳ�ֵ **/
        const int minDisparity;
        /** \����Ӳ�ֵ **/
        const int maxDisparity;
        /** \��С���ֵ **/
        const float minDepth;
        /** \������ֵ **/
        const float maxDepth;
        /** \block�ߴ� **/
        const dim3 block;
    };
}
#endif // Restruction_GPU_H
