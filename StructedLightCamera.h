/**
 * @file StructedLightCamera.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  结构光相机（双目+色彩）
 *         解相算法：1、三步五位互补格雷码   可选：CPU
 *                 2、四步五位互补格雷码   可选：CPU | GPU
 *                 3、位移格雷码+时间复用策略（3+1，3+3） 可选：GPU
 *                 4、分区间相位展开+时间复用策略（3+1,3+1,3+1,3+1) 可选：GPU
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
    /** \算法类型 **/
    enum AlgorithmType{
        ThreeStepFiveGrayCode = 0,//3+5 CounterGrayCode
        FourStepSixGrayCode = 1,//4+6 CunterGrayCode
        DevidedSpaceTimeMulUsed = 2,//3+1,3+1,3+1,3+1
        ShiftGrayCodeTimeMulUsed = 3//3+1,3+3
    };
    /** \加速方法 **/
    enum AcceleratedMethod{
        CPU = 0,
        GPU = 1
    };
    /**
     * @brief 构造函数
     * @param infoCalibraion 输入，标定信息
     * @param algorithmType 输入，算法类型
     * @param acceleratedMethod 输入，加速方法
     * @param params 输入，重建器控制参数
     * @param leftRefImg 输入，左参考绝对相位
     * @param rightRefImg 输入，右参考绝对相位
     */
    StructedLightCamera(const Info& infoCalibraion, AlgorithmType algorithmType = AlgorithmType::ShiftGrayCodeTimeMulUsed, AcceleratedMethod acceleratedMethod = AcceleratedMethod::CPU, const RestructorType::RestructParamater params = RestructorType::RestructParamater(),
                        const cv::Mat& leftRefImg = cv::Mat(0,0,CV_32FC1), const cv::Mat& rightRefImg = cv::Mat(0,0,CV_32FC1));
    /**
     * @brief 获取深度纹理图
     * @param depthImg 输入/输出，深度图
     * @param colorImg 输入/输出，纹理图
     */
    void getOneFrame(std::vector<cv::Mat>& depthImg,std::vector<cv::Mat>& colorImg);
#ifdef CUDA
    /**
     * @brief 获取深度纹理图
     * @param depthImg 输入/输出，深度图
     * @param colorImg 输入/输出，纹理图
     */
    void getOneFrame(std::vector<cv::cuda::GpuMat>& depthImg, std::vector<cv::cuda::GpuMat>& colorImg);
#endif
    /**
     * @brief 设置相机曝光时间
     * @param grayExposureTime 输入，灰度相机曝光时间
     * @param colorExposureTime 输入，彩色相机曝光时间
     */
    void setExposureTime(const int grayExposureTime, const int colorExposureTime);
    /**
     * @brief 关闭相机
     */
    void closeCamera();
private:
    /**
     * @brief 校正图片
     * @param src 输入，原图片
     * @param remap_x 输入，X方向映射矩阵
     * @param remap_y 输入，Y方向映射矩阵
     * @param outImg 输入/输出，校正后的图片
     */
    void remapImg(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& remap_x, const cv::cuda::GpuMat& remap_y, cv::cuda::GpuMat& outImg);
    /** \标定信息 **/
    const Info& calibrationInfo;
    /** \左相机解相器 **/
    PhaseSolverType::PhaseSolver* phaseSolverLeft;
    /** \右相机解相器 **/
    PhaseSolverType::PhaseSolver* phaseSolverRight;
    /** \重建器 **/
    RestructorType::Restructor* restructor;
    /** \相机控制器 **/
    CameraControl* camera;
    /** \解相算法 **/
    AlgorithmType algorithmType;
    /** \加速方法 **/
    AcceleratedMethod acceleratedMethod;
    /** \X方向映射矩阵-L **/
    cv::Mat remap_x_L;
    /** \Y方向映射矩阵-L **/
    cv::Mat remap_y_L;
    /** \X方向映射矩阵-R **/
    cv::Mat remap_x_R;
    /** \Y方向映射矩阵-R **/
    cv::Mat remap_y_R;
#ifdef CUDA
    /** \左解相非阻塞流 **/
    cudaStream_t stream_solLeft;
    /** \右解相非阻塞流 **/
    cudaStream_t stream_solRight;
    /** \X方向映射矩阵-CUDA-L **/
    cv::cuda::GpuMat remap_x_deice_L;
    /** \X方向映射矩阵-CUDA-L **/
    cv::cuda::GpuMat remap_y_deice_L;
    /** \X方向映射矩阵-CUDA-R **/
    cv::cuda::GpuMat remap_x_deice_R;
    /** \X方向映射矩阵-CUDA-R **/
    cv::cuda::GpuMat remap_y_deice_R;
#endif
};
