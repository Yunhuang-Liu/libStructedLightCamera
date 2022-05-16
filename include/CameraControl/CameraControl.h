/**
 * @file CammeraControl.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  �ṹ�����������
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
     * @brief Ĭ�Ϲ��캯��
     * @param projectorModuleType ���룬ͶӰ������
     */
    CameraControl(const DLPC34XX_ControllerDeviceId_e projectorModuleType);
    /**
      * @brief ��ȡһ֡ͼƬ
      * @param imgsOneFrame ���룬��ȡ����ԭʼͼƬ
      */
     void getOneFrameImgs(RestructedFrame& imgsOneFrame);
     /**
      * @brief ���ò���ͼƬ����
      * @param GrayImgsNum  ���룬�Ҷ������������
      * @param ColorImgsNum ���룬��ɫ�����������
      */
     void setCaptureImgsNum(const int GrayImgsNum,const int ColorImgsNum);
     /**
      * @brief ���ع̼�
      * @param firmwarePath ���룬�̼���ַ
      */
     void loadFirmware(const std::string firmwarePath);
     /**
      * @brief ����ɫ�������Ϊ����������һ��
      */
     void triggerColorCameraSoftCaputure();
     /**
      * @brief ��������ع�ʱ��
      * @param grayExposure ���룬�Ҷ�����ع�ʱ��
      * @param colorExposure ���룬��ɫ����ع�ʱ��
      */
     void setCameraExposure(const int grayExposure, const int colorExposure);
     /**
      * @brief �ر���� 
      */
     void closeCamera();
private:
    /** \����� **/
    CammeraUnilty* cameraLeft;
    /** \����� **/
    CammeraUnilty* cameraRight;
    /** \��ɫ��� **/
    CammeraUnilty* cameraColor;
    /** \ͶӰ�� **/
    ProjectorControl* projector;
};

#endif
