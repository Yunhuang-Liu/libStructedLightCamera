/**
 * @file StructedLightCamera.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  �ṹ�������˫Ŀ+ɫ�ʣ�
 *         �����㷨��1��������λ����������   ��ѡ��CPU
 *                 2���Ĳ���λ����������   ��ѡ��CPU | GPU
 *                 3��λ�Ƹ�����+ʱ�临�ò��ԣ�3+1��3+3�� ��ѡ��GPU
 *                 4����������λչ��+ʱ�临�ò��ԣ�3+1,3+1,3+1,3+1) ��ѡ��GPU
 * @version 0.1
 * @date 2022-5-9
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "./CameraControl/CameraControl.h"
#include "./Restructor/FourStepSixGrayCodeMaster_CPU.h"
#include "./Restructor/ThreeStepFiveGrayCodeMaster_CPU.h"
#include "./Restructor/Restructor_CPU.h"

#ifdef CUDA
#include "./Restructor/Restructor_GPU.h"
#include "./Restructor/FourStepSixGrayCodeMaster_GPU.h"
#include "./Restructor/DividedSpaceTimeMulUsedMaster_GPU.h"
#include "./Restructor/ShiftGrayCodeUnwrapMaster_GPU.h"
#endif

class StructedLightCamera{
public:
    /** \�㷨���� **/
    enum AlgorithmType{
        ThreeStepFiveGrayCode = 0,//3+5 CounterGrayCode
        FourStepSixGrayCode = 1,//4+6 CunterGrayCode
        DevidedSpaceTimeMulUsed = 2,//3+1,3+1,3+1,3+1
        ShiftGrayCodeTimeMulUsed = 3//3+1,3+3
    };
    /** \���ٷ��� **/
    enum AcceleratedMethod{
        CPU = 0,
        GPU = 1
    };
    /**
     * @brief ���캯��
     * @param infoCalibraion ���룬�궨��Ϣ
     * @param algorithmType ���룬�㷨����
     * @param acceleratedMethod ���룬���ٷ���
     * @param params ���룬�ؽ������Ʋ���
     * @param leftRefImg ���룬��ο�������λ
     * @param rightRefImg ���룬�Ҳο�������λ
     */
    StructedLightCamera(const Info& infoCalibraion, AlgorithmType algorithmType = AlgorithmType::ShiftGrayCodeTimeMulUsed, AcceleratedMethod acceleratedMethod = AcceleratedMethod::CPU, const RestructorType::RestructParamater params = RestructorType::RestructParamater(),
                        const cv::Mat& leftRefImg = cv::Mat(0,0,CV_32FC1), const cv::Mat& rightRefImg = cv::Mat(0,0,CV_32FC1));
    /**
     * @brief ��ȡ�������ͼ
     * @param depthImg ����/��������ͼ
     * @param colorImg ����/���������ͼ
     */
    void getOneFrame(std::vector<cv::Mat>& depthImg,std::vector<cv::Mat>& colorImg);
#ifdef CUDA
    /**
     * @brief ��ȡ�������ͼ
     * @param depthImg ����/��������ͼ
     * @param colorImg ����/���������ͼ
     */
    void getOneFrame(std::vector<cv::cuda::GpuMat>& depthImg, std::vector<cv::cuda::GpuMat>& colorImg);
#endif
    /**
     * @brief ��������ع�ʱ��
     * @param grayExposureTime ���룬�Ҷ�����ع�ʱ��
     * @param colorExposureTime ���룬��ɫ����ع�ʱ��
     */
    void setExposureTime(const int grayExposureTime, const int colorExposureTime);
    /**
     * @brief �ر����
     */
    void closeCamera();
private:
    /**
     * @brief У��ͼƬ
     * @param src ���룬ԭͼƬ
     * @param remap_x ���룬X����ӳ�����
     * @param remap_y ���룬Y����ӳ�����
     * @param outImg ����/�����У�����ͼƬ
     */
    void remapImg(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& remap_x, const cv::cuda::GpuMat& remap_y, cv::cuda::GpuMat& outImg);
    /** \�궨��Ϣ **/
    const Info& calibrationInfo;
    /** \����������� **/
    PhaseSolverType::PhaseSolver* phaseSolverLeft;
    /** \����������� **/
    PhaseSolverType::PhaseSolver* phaseSolverRight;
    /** \�ؽ��� **/
    RestructorType::Restructor* restructor;
    /** \��������� **/
    CameraControl* camera;
    /** \�����㷨 **/
    AlgorithmType algorithmType;
    /** \���ٷ��� **/
    AcceleratedMethod acceleratedMethod;
    /** \X����ӳ�����-L **/
    cv::Mat remap_x_L;
    /** \Y����ӳ�����-L **/
    cv::Mat remap_y_L;
    /** \X����ӳ�����-R **/
    cv::Mat remap_x_R;
    /** \Y����ӳ�����-R **/
    cv::Mat remap_y_R;
#ifdef CUDA
    /** \������������ **/
    cudaStream_t stream_solLeft;
    /** \�ҽ���������� **/
    cudaStream_t stream_solRight;
    /** \X����ӳ�����-CUDA-L **/
    cv::cuda::GpuMat remap_x_deice_L;
    /** \X����ӳ�����-CUDA-L **/
    cv::cuda::GpuMat remap_y_deice_L;
    /** \X����ӳ�����-CUDA-R **/
    cv::cuda::GpuMat remap_x_deice_R;
    /** \X����ӳ�����-CUDA-R **/
    cv::cuda::GpuMat remap_y_deice_R;
#endif
};
