/**
 * @file CammeraControl.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  结构光相机控制类
 * @version 0.1
 * @date 2022-5-9
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef CameraControl_H
#define CameraControl_H

#include "./CameraUtility/CammeraUnilty.h"
#include "./ProjectorSDK/ProjectorControl.h"

struct RestructedFrame{
public:
    RestructedFrame() {};
    RestructedFrame(std::vector<cv::Mat>& leftImgs_, std::vector<cv::Mat>& rightImgs_, std::vector<cv::Mat>& colorImgs_) :
    leftImgs(leftImgs_), rightImgs(rightImgs_), colorImgs(colorImgs_){}
    std::vector<cv::Mat> leftImgs;
    std::vector<cv::Mat> rightImgs;
    std::vector<cv::Mat> colorImgs;
};

class CameraControl{
public:
    /**
     * @brief 默认构造函数
     * @param projectorModuleType 输入，投影仪类型
     */
    CameraControl(const DLPC34XX_ControllerDeviceId_e projectorModuleType);
    /**
      * @brief 获取一帧图片
      * @param imgsOneFrame 输入，获取到的原始图片
      */
     void getOneFrameImgs(RestructedFrame& imgsOneFrame);
     /**
      * @brief 设置捕获图片数量
      * @param GrayImgsNum  输入，灰度相机捕获张数
      * @param ColorImgsNum 输入，彩色相机捕获张数
      */
     void setCaptureImgsNum(const int GrayImgsNum,const int ColorImgsNum);
     /**
      * @brief 加载固件
      * @param firmwarePath 输入，固件地址
      */
     void loadFirmware(const std::string firmwarePath);
     /**
      * @brief 将彩色相机设置为软触发并触发一次
      */
     void triggerColorCameraSoftCaputure();
     /**
      * @brief 设置相机曝光时间
      * @param grayExposure 输入，灰度相机曝光时间
      * @param colorExposure 输入，彩色相机曝光时间
      */
     void setCameraExposure(const int grayExposure, const int colorExposure);
     /**
      * @brief 关闭相机 
      */
     void closeCamera();
private:
    /** \左相机 **/
    CammeraUnilty* cameraLeft;
    /** \右相机 **/
    CammeraUnilty* cameraRight;
    /** \彩色相机 **/
    CammeraUnilty* cameraColor;
    /** \投影仪 **/
    ProjectorControl* projector;
};

#endif
