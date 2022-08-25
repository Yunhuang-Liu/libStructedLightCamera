/**
 * @file Restructor_CPU_GrayPhase.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  CPU�ؽ���(SIMD:AVX(256bit),���߳�)
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef Restructor_CPU_H
#define Restructor_CPU_H
#include "Restructor.h"
#include <immintrin.h>
#include "MatrixsInfo.h"
#include <limits>

namespace RestructorType {
    class Restructor_CPU : public Restructor{
        public:
        /**
             * @brief ���캯��
             * @param calibrationInfo ���룬�궨��Ϣ
             * @param minDisparity ���룬��С�Ӳ�
             * @param maxDisparity ���룬����Ӳ�
             * @param minDepth ���룬��С���ֵ
             * @param maxDepth ���룬������ֵ
             * @param threads ���룬�߳���
             */
            Restructor_CPU(const Info& calibrationInfo, const int minDisparity = -500, const int maxDisparity = 500, 
                           const float minDepth = 170, const float maxDepth = 220, const int threads = 16);
            /**
             * @brief ��������
             */
            ~Restructor_CPU();
            /**
             * @brief �ؽ�
             * @param leftAbsImg ���룬�������λ
             * @param rightAbsImg ���룬�Ҿ�����λ
             * @param depthImgOut ����/��������ͼ
             */
            void restruction(const cv::Mat& leftAbsImg, const cv::Mat& rightAbsImg, cv::Mat& depthImgOut, const bool isColor = false) override;
        protected:
            /**
             * @brief ӳ���������
             * @param leftAbsImg ���룬�������λ
             * @param rightAbsImg ���룬�Ҿ�����λ
             * @param colorImg ���룬ԭʼ��ɫͼƬ
             * @param depthImgOut ����/��� ���ͼ
             * @param colorImgOut ����/��� ����ͼ
             */
            void getDepthColorMap(const cv::Mat& leftAbsImg, const cv::Mat& rightAbsImg,
                                  cv::Mat& depthImgOut, const bool isColor = false);
        private:
            #ifdef CUDA
            //GPU�˺���
            void download(const int index, cv::cuda::GpuMat& depthImg) {};
            void restruction(const cv::cuda::GpuMat& leftAbsImg, const cv::cuda::GpuMat& rightAbsImg,
                const int sysIndex, cv::cuda::Stream& stream, const bool isColor = false) override {}
            #endif
            /** \�洢�߳��� **/
            std::mutex mutexMap;
            /** \�궨��Ϣ **/
            const Info& calibrationInfo;
            /** \ʹ���߳��� **/
            const int threads;
            /** \��С�Ӳ�ֵ **/
            int minDisparity;
            /** \����Ӳ�ֵ **/
            int maxDisparity;
            /** \��С��� **/
            const float minDepth;
            /** \������ **/
            const float maxDepth;
            /** \��ͶӰֱ�ӻ�ȡ����(���߳���ں���) **/
            void thread_DepthColorMap(const cv::Mat& leftAbsImg, const cv::Mat& righAbstImg,
                                       cv::Mat& depthImgOut, const cv::Point2i region, const bool isColor = false);
    };
}
#endif // !Restructor_CPU_H
