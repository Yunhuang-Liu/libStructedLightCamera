/**
 * @file cammeraUnilty.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  相机工具类：值得一提的是大华和海康相机皆可使用，采用同一标准。
 * @version 0.1
 * @date 2021-12-10
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef CAMERA_CAMMERUNILTY_H_
#define CAMERA_CAMMERUNILTY_H_

#include <device/camera/IMVApi.h>

#include <vector>

#include <opencv2/opencv.hpp>

/** @brief 结构光库 */
namespace sl {
    /** @brief 设备控制库 */
    namespace device {
        /** @brief 帧信息 */
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
            unsigned char *m_pImageBuf;
            int m_nBufferSize;
            int m_nWidth;
            int m_nHeight;
            IMV_EPixelType m_ePixelType;
            int m_nPaddingX;
            int m_nPaddingY;
            uint64_t m_nTimeStamp;
        };
        /** @brief 相机控制类 **/
        class CammeraUnilty {
        public:
            /** @brief 相机类别 **/
            enum CameraType {
                LeftCamera = 0, //左相机
                RightCamera = 1,//右相机
                ColorCamera = 2,//彩色相机
            };
            /** @brief 枚举触发方式 **/
            enum ETrigType {
                trigContinous = 0,// 连续拉流 | continue grabbing
                trigSoftware = 1, // 软件触发 | software trigger
                trigLine = 2,     // 外部触发	| external trigger
            };
            explicit CammeraUnilty();
            ~CammeraUnilty();
            /**
             * @brief 打开相机
             * 
             * @return true：成功，false：失败
             */
            bool CameraOpen(void);
            /**
             * @brief 关闭相机
             * 
             * @return true：成功，false：失败
             */
            bool CameraClose(void);
            /**
             * @brief 开始拉流
             * 
             * @return true：成功，false：失败
             */
            bool CameraStart(void);
            /**
             * @brief 关闭相机
             * 
             * @return true：成功，false：失败
             */
            bool CameraStop(void);
            /**
             * @brief 切换采集方式、触发方式 （连续采集、外部触发、软件触发）
             * 
             * @param trigType  输入，触发方式
             * @return true：成功，false：失败
             */
            bool CameraChangeTrig(ETrigType trigType = trigContinous);
            /**
             * @brief 执行一次软触发
             * 
             * @return true：成功，false：失败
             */
            bool ExecuteSoftTrig(void);
            /**
             * @brief 设置曝光
             * 
             * @param dExposureTime 输入，曝光时间
             * @return true：成功，false：失败
             */
            bool SetExposeTime(double dExposureTime);
            /**
             * @brief 查询是否硬触发
             * 
             * @return true：硬触发，false：软触发
             */
            bool isTriggerLine();
            /**
             * @brief 设置增益
             * 
             * @param dGainRaw 输入，增益值
             * @return true：成功，false：失败
             */
            bool SetAdjustPlus(double dGainRaw);
            /**
             * @brief 设置当前相机
             * 
             * @param strKey 输入，相机键值
             */
            void SetCamera(const std::string &strKey);
            /**
             * @brief 设置ROI
             * 
             * @param width   输入，幅面宽度
             * @param height  输入，幅面高度
             * @param offsetX 输入，幅面起点自左上角X偏移量
             * @param offsetY 输入，幅面起点自左上角Y偏移量
             */
            void setROI(const int width, const int height,
                        const int offsetX, const int offsetY);
            /**
             * @brief 清除缓存
             */
            void clearBuffer();
            /**
             * @brief 设置自动曝光
             * 
             * @param isAutoExp 输入，是否自动曝光
             */
            void setAutoExposure(const bool isAutoExp);
            /**
             * @brief 设置图片格式
             * 
             * @param pixelFormat 输入，图片格式
             */
            void setPixelFormat(const std::string pixelFormat);
            /**
             * @brief 设置自动白平衡
             * 
             * @param pixelFormat 输入，是否自动白平衡
             */
            void setAutoWhiteBlance(const bool isAutoBlance);
            /**
             * @brief 设置亮度
             * 
             * @param brightness 输入，亮度值
             */
            void setBrightness(const int brightness);
            /**
             * @brief 设置白平衡R、G、B增益值
             * 
             * @param r 输入，红色值
             * @param g 输入，白色值
             * @param b 输入，蓝色值
             */
            void setWhiteBlanceRGB(float r, float g, float b);
            /** \是否彩色相机 **/
            CameraType cameraType;
            /** \相机句柄 **/
            IMV_HANDLE m_devHandle = nullptr;
            /** \设置的曝光时间 **/
            int exposureTime;
            /** \图片队列 **/
            std::queue<cv::Mat> imgQueue;
        private:
            /** \当前相机key **/
            std::string m_currentCameraKey;
        };
    }// namespace device
}// namespace sl
#endif // CAMERA_CAMMERUNILTY_H_
