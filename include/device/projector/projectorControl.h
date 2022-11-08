/**
 * @file projectorControl.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  投影仪控制类
 * @version 0.1
 * @date 2022-5-9
 *
 * @copyright Copyright (c) 2022
 *
 */


#ifndef PROJECTOR_PROJECTORCONTROL_H_
#define PROJECTOR_PROJECTORCONTROL_H_

#include <device/projector/dlpc_common.h>
#include <device/projector/dlpc34xx.h>
#include <device/projector/dlpc347x_internal_patterns.h>
#include <device/projector/cypress_i2c.h>
#include <device/projector/API.h>
#include <device/projector/usb.h>

#include <string>
#include <iostream>
#include <thread>

#define FLASH_WRITE_BLOCK_SIZE            1024
#define FLASH_READ_BLOCK_SIZE             256

#define MAX_WRITE_CMD_PAYLOAD             (FLASH_WRITE_BLOCK_SIZE + 8)
#define MAX_READ_CMD_PAYLOAD              (FLASH_READ_BLOCK_SIZE  + 8)

/** \write buffer **/
static uint8_t s_WriteBuffer[MAX_WRITE_CMD_PAYLOAD];
/** \write buffer **/
static uint8_t s_ReadBuffer[MAX_READ_CMD_PAYLOAD];
/** \file pointer **/
static FILE* s_FilePointer;

/** @brief 结构光库 */
namespace sl {
    /** @brief 设备控制库 */
    namespace device {
        /**
         * @brief 投影仪控制类
         * @note 应当注意的是，DLP3010将得到全面的支持，无需GUI进行辅助，DLP6500的需求将迫使你不得不使用GUI事先进行图片的烧入 
         */
        class ProjectorControl {
        public:
            /**
             * @brief 构造函数
             * @param projectorType 输入，投影仪类别
             */
            ProjectorControl(const DLPC34XX_ControllerDeviceId_e projectorType);
            /** 
             * @brief 构造函数
             * @param projectorType 输入，投影仪类别
             */
            ProjectorControl(const int numLutEntries);
            /**
             * @brief 开始投影
             * @param isContinues 输入，是否连续投影
             */
            void projecte(const bool isContinues);
            /**
             * @brief 停止投影
             */
            void stopProject();
            /**
             * @brief 加载固件
             * @param firmWareAdress 输入，固件文件地址
             */
            void LoadFirmware(const std::string firmWareAdress);
            /** \投影图片张数 **/
            const int elementSize;
        private:
            /**
             * @brief 初始化I2C
             */
            void InitConnectionAndCommandLayer();
            /**
             * @brief 等待
             * @param Seconds 输入，等待时长
             */
            void WaitForSeconds(uint32_t Seconds);
            /**
             * @brief 从闪存加载图片
             */
            void LoadPatternOrderTableEntryfromFlash();
            /** \是否为DLPC900控制芯片 **/
            const bool isDLPC900;
        };
    }// namespace device
}// namespace sl
#endif //PROJECTOR_PROJECTORCONTROL_H_
