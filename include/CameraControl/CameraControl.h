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
    RestructedFrame(std::vector<cv::Mat>& leftImgs_, std::vector<cv::Mat>& rightImgs_) : leftImgs(leftImgs_), rightImgs(rightImgs_){}
    std::vector<cv::Mat> leftImgs;
    std::vector<cv::Mat> rightImgs;
    std::vector<cv::Mat> colorImgs;
};

class CameraControl{
public:
    enum CameraUsedState {
        LeftGrayRightGray = 0,
        LeftColorRightGray = 1,
        LeftGrayRightGrayExColor = 2,
    };
    /**
     * @brief Ĭ�Ϲ��캯��
     * 
     * @param projectorModuleType ���룬ͶӰ������
     * @param state ���룬�������״̬
     */
    CameraControl(const DLPC34XX_ControllerDeviceId_e projectorModuleType, CameraUsedState state = LeftGrayRightGrayExColor);
    /**
     * @brief Ĭ�Ϲ��캯��
     *
     * @param numLutEntries ���룬DLP6500ͶӰ������
     *                      *** Warning Begin ***
     *                      ��ע�⣬����ʹ�ñ�SDK������ʹ��GUI����ͼƬ
     *                      *** Warning End ***
     * @param state ���룬�������״̬
     */
    CameraControl(const int numLutEntries, CameraUsedState state = LeftGrayRightGrayExColor);
    /**
      * @brief ��ȡһ֡ͼƬ
      * 
      * @param imgsOneFrame ���룬��ȡ����ԭʼͼƬ
      */
     void getOneFrameImgs(RestructedFrame& imgsOneFrame);
     /**
      * @brief ���ò���ͼƬ����
      * 
      * @param GrayImgsNum  ���룬�Ҷ������������
      * @param ColorImgsNum ���룬��ɫ�����������
      */
     void setCaptureImgsNum(const int GrayImgsNum,const int ColorImgsNum);
     /**
      * @brief ���ع̼�
      * 
      * @param firmwarePath ���룬�̼���ַ
      */
     void loadFirmware(const std::string firmwarePath);
     /**
      * @brief ����ɫ�������Ϊ����������һ��
      */
     void triggerColorCameraSoftCaputure();
     /**
      * @brief ��������ع�ʱ��
      * 
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
    /** \�������״̬ **/
    CameraUsedState cameraUsedState;
};

#endif
