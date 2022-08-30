/**
 * @file CammeraUnilty.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  相机工具类：值得一提的是大华和海康相机皆可使用，采用同一标准。
 * @version 0.1
 * @date 2021-12-10
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef CAMERACONTROL_CAMMERUNILTY_H
#define CAMERACONTROL_CAMMERUNILTY_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "../CameraSDK/IMVApi.h"

// 状态栏统计信息 
// Status bar statistics
struct FrameStatInfo {
    unsigned int	m_nFrameSize;   // 帧大小, 单位: 字节 | frame size ,
                                    // length :byte
    uint64_t		m_nPassTime;    // 接收到该帧时经过的纳秒数 |  
                                    // The number of nanoseconds passed when
                                    // the frame was received
    FrameStatInfo(unsigned int nSize, uint64_t nTime) : 
        m_nFrameSize(nSize), m_nPassTime(nTime) {}
};

// 帧信息 
// frame imformation
class CFrameInfo {
public:
    CFrameInfo() {
        m_pImageBuf = NULL;
        m_nBufferSize = 0;
        m_nWidth = 0;
        m_nHeight = 0;
        m_ePixelType = gvspPixelMono8;
        m_nPaddingX = 0;
        m_nPaddingY = 0;
        m_nTimeStamp = 0;
    }

    ~CFrameInfo() {
    }

public:
    unsigned char*	m_pImageBuf;
    int				m_nBufferSize;
    int				m_nWidth;
    int				m_nHeight;
    IMV_EPixelType	m_ePixelType;
    int				m_nPaddingX;
    int				m_nPaddingY;
    uint64_t		m_nTimeStamp;
};

class CammeraUnilty {
public:
    /** \相机类别 **/
    enum CameraType {
        LeftCamera = 0,
        RightCamera = 1,
        ColorCamera = 2,
    };
    /** \枚举触发方式 **/
    enum ETrigType {
        trigContinous = 0,	// 连续拉流 | continue grabbing
        trigSoftware = 1,	// 软件触发 | software trigger
        trigLine = 2,		// 外部触发	| external trigger
    };
    explicit CammeraUnilty();
    ~CammeraUnilty();
    /** \打开相机 **/
    bool CameraOpen(void);
    /** \关闭相机 **/
    bool CameraClose(void);
    /** \开始采集 **/
    bool CameraStart(void);
    /** \执行一次软触发 **/
    bool CameraStop(void);
    /** \切换采集方式、触发方式 （连续采集、外部触发、软件触发） **/
    bool CameraChangeTrig(ETrigType trigType = trigContinous);
    /** \执行一次软触发 **/
    bool ExecuteSoftTrig(void);
    /** \设置曝光 **/
    bool SetExposeTime(double dExposureTime);
    /** \查询是否硬触发——0：软触发、1：硬触发 **/
    bool isTriggerLine();
    /** \设置增益 **/
    bool SetAdjustPlus(double dGainRaw);
    /** \设置当前相机 **/
    void SetCamera(const std::string& strKey);
    /** \设置ROI **/
    void setROI(const int width, const int height, 
        const int offsetX, const int offsetY);
    /** \清除Buufer **/
    void clearBuffer();
    /** \设置自动曝光 **/
    void setAutoExposure(const bool isAutoExp);
    /** \设置图片格式 **/
    void setPixelFormat(const std::string pixelFormat);
    /** \设置自动白平衡 **/
    void setAutoWhiteBlance(const bool isAutoBlance);
    /** \设置亮度 **/
    void setBrightness(const int brightness);
    /** \设置白平衡R、G、B增益值 **/
    void setWhiteBlanceRGB(float r,float g,float b);		
     /** \所有图片 **/
    std::vector<cv::Mat> imgs;
    /** \图片索引 **/
    int index;
    /** \是否彩色相机 **/
    CameraType cameraType;
    /** \相机句柄 **/
    IMV_HANDLE  m_devHandle = nullptr;		
private:
    /** \当前相机key **/
    std::string m_currentCameraKey;			
};

#endif // CAMERACONTROL_CAMMERUNILTY_H
