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

#ifndef PROJECTORSDK_CameraControl_H
#define PROJECTORSDK_CameraControl_H

#include <CameraControl/CameraUtility/CammeraUnilty.h>
#include <CameraControl/ProjectorSDK/ProjectorControl.h>

/** @brief 结构光库 */
namespace SL {
    /** @brief 设备控制库 */
    namespace Device {
        /** @brief 重建帧 */
        struct RestructedFrame {
        public:
            RestructedFrame(){};
            /**
             * @brief                带有彩色图片的构造函数
             * 
             * @param leftImgs_      输入，左相机图片
             * @param rightImgs_     输入，右相机图片
             * @param colorImgs_     输入，彩色相机图片
             */
            RestructedFrame(std::vector<cv::Mat> &leftImgs_,
                            std::vector<cv::Mat> &rightImgs_,
                            std::vector<cv::Mat> &colorImgs_) {
                leftImgs.resize(leftImgs_.size());
                rightImgs.resize(rightImgs_.size());
                colorImgs.resize(colorImgs_.size());
                for (int i = 0; i < leftImgs.size(); i++) {
                    leftImgs[i] = leftImgs_[i];
                }
                for (int i = 0; i < rightImgs_.size(); i++) {
                    rightImgs[i] = rightImgs_[i];
                }
                for (int i = 0; i < colorImgs_.size(); i++) {
                    colorImgs[i] = colorImgs_[i];
                }
            }
            /**
             * @brief                带有彩色图片的构造函数
             * 
             * @param leftImgs_      输入，左相机图片
             * @param rightImgs_     输入，右相机图片
             */
            RestructedFrame(std::vector<cv::Mat> &leftImgs_,
                            std::vector<cv::Mat> &rightImgs_) {
                leftImgs.resize(leftImgs_.size());
                rightImgs.resize(rightImgs_.size());
                for (int i = 0; i < leftImgs.size(); i++) {
                    leftImgs[i] = leftImgs_[i];
                }
                for (int i = 0; i < rightImgs_.size(); i++) {
                    rightImgs[i] = rightImgs_[i];
                }
            }
            //左相机拍摄图片
            std::vector<cv::Mat> leftImgs;
            //右相机拍摄图片
            std::vector<cv::Mat> rightImgs;
            //彩色相机拍摄图片
            std::vector<cv::Mat> colorImgs;
        };

        /** @brief 结构光相机控制类 */
        class CameraControl {
        public:
            /** @brief 相机使用状态 **/
            enum CameraUsedState {
                LeftGrayRightGray = 0,        //左灰度右灰度
                LeftColorRightGray = 1,       //左彩色右灰度
                LeftGrayRightGrayExColor = 2, //左灰度右灰度额外彩色
            };
            /**
             * @brief 默认构造函数
             * 
             * @param projectorModuleType 输入，投影仪类型
             * @param state 输入，相机配置状态
             */
            CameraControl(const DLPC34XX_ControllerDeviceId_e projectorModuleType,
                          CameraUsedState state = LeftGrayRightGrayExColor);
            /**
             * @brief 默认构造函数
             *
             * @param numLutEntries 输入，DLP6500投影张数，
             * @warning 请注意，若需使用本SDK，需事使用GUI加载图片
             * @param state 输入，相机配置状态
             */
            CameraControl(const int numLutEntries,
                          CameraUsedState state = LeftGrayRightGrayExColor);
            /**
             * @brief 获取一帧图片
             * 
             * @param imgsOneFrame 输入，获取到的原始图片
             */
            void getOneFrameImgs(RestructedFrame &imgsOneFrame);
            /**
             * @brief 设置捕获图片数量
             * 
             * @param GrayImgsNum  输入，灰度相机捕获张数
             * @param ColorImgsNum 输入，彩色相机捕获张数
             */
            void setCaptureImgsNum(const int GrayImgsNum, const int ColorImgsNum);
            /**
             * @brief 加载固件
             * 
             * @param firmwarePath 输入，固件地址
             */
            void loadFirmware(const std::string firmwarePath);
            /**
             * @brief 将彩色相机设置为软触发并触发一次
             */
            void triggerColorCameraSoftCaputure();
            /**
             * @brief 设置相机曝光时间
             * 
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
            CammeraUnilty *cameraLeft;
            /** \右相机 **/
            CammeraUnilty *cameraRight;
            /** \彩色相机 **/
            CammeraUnilty *cameraColor;
            /** \投影仪 **/
            ProjectorControl *projector;
            /** \相机配置状态 **/
            CameraUsedState cameraUsedState;
        };
    }// namespace Device
}// namespace SL
#endif // PROJECTORSDK_CameraControl_H
